import gspread.utils
import schwab
import configparser
import httpx
import argparse
import pandas as pd
import pandas_ta as ta
import datetime as dt
import numpy as np
import json
import os
import time
import gspread

from tabulate import tabulate
from datetime import timedelta
from gspread_formatting import color, CellFormat, format_cell_range
from google.oauth2.service_account import Credentials
from authlib.integrations.base_client import OAuthError
from datetime import datetime

VERSION = '2024-09-28'

print(f'Version: {VERSION}')

# Read configurations from config file
config = configparser.ConfigParser()
config.read('schwab_config.ini')

api_key = config['schwab']['api_key']
app_secret = config['schwab']['app_secret']
redirect_uri = config['schwab']['redirect_uri']
token_path = config['schwab']['token_path']
account_number = config['schwab']['account_number']

open_interest_df = None
schwab_client = None
yesterday_ohlc = {}

def get_the_next_friday(given_date):
    # If the given date is a Friday, return it
    if given_date.weekday() == 4:  # Friday is represented by 4
        return given_date.strftime('%Y-%m-%d')
    
    # Calculate the number of days until the next Friday
    days_until_friday = (4 - given_date.weekday() + 7) % 7
    next_friday_date = given_date + timedelta(days=days_until_friday)
    
    # Return the next Friday date as a string in the format %Y-%m-%d
    return next_friday_date.strftime('%Y-%m-%d')

def initialize_schwab_client(reauthorize: bool=False):
    """Initialize Schwab client and handle authentication."""
    try:
        if not reauthorize:
            return schwab.auth.client_from_token_file(token_path, api_key, app_secret)
        else:
            return schwab.auth.client_from_manual_flow(api_key, app_secret, redirect_uri, token_path)
    except FileExistsError:
        return schwab.auth.client_from_manual_flow(api_key, app_secret, redirect_uri, token_path)
    except FileNotFoundError:
        return schwab.auth.client_from_manual_flow(api_key, app_secret, redirect_uri, token_path)

def calculate_open_interest(data_map, option_type):
    open_interest = {}
    for exp_date, strikes in data_map.items():
        for strike_price, options in strikes.items():
            option_info = options[0]  # Assuming each list contains one option data dict
            strike = float(option_info['strikePrice'])
            oi = option_info['openInterest']
            if strike not in open_interest:
                open_interest[strike] = {'CALL': 0, 'PUT': 0}
            open_interest[strike][option_type] += oi
    return open_interest

def calculate_max_pain(open_interest_df):
    max_pain = 0
    min_total_loss = float('inf')
    
    for strike in open_interest_df['Strike']:
        total_loss = 0
        for _, row in open_interest_df.iterrows():
            if row['Strike'] <= strike:
                # For calls, loss is max(0, strike price - option's strike price)
                total_loss += row['Calls'] * max(0, strike - row['Strike'])
            else:
                # For puts, loss is max(0, option's strike price - strike price)
                total_loss += row['Puts'] * max(0, row['Strike'] - strike)
        
        if total_loss < min_total_loss:
            min_total_loss = total_loss
            max_pain = strike
    
    return max_pain

def calculate_adr(df: pd.DataFrame, days: int=5):
    adr = ta.sma(df['high'] - df['low'], length=days)
    return round(adr.iloc[-1], 2)

def get_ohlc_history(ticker, days: int=5) -> pd.DataFrame:
    global schwab_client

    start_date = dt.date.today() - dt.timedelta(days=days*2)
    end_date = dt.date.today() - dt.timedelta(days=1)
    start_datetime = dt.datetime.combine(start_date, dt.time())
    end_datetime = dt.datetime.combine(end_date, dt.time())
    resp = schwab_client.get_price_history_every_day(ticker, start_datetime=start_datetime, end_datetime=end_datetime)  
    if resp.status_code in {httpx.codes.OK, httpx.codes.CREATED}:
        data = resp.json()        
        candles = data.get('candles')
        if len(candles) < days:
            print("Not enough data!")
            return None
               
        df = pd.DataFrame(candles)
        
        # Convert datetime column from milliseconds to seconds
        df['datetime'] = df['datetime'] / 1000

        # Convert datetime column to datetime object
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')

        # Convert datetime column to string date
        df['date_string'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Set the date_string column as the index
        df.set_index('date_string', inplace=True)

        # Drop the original datetime column if not needed
        df.drop(columns=['datetime'], inplace=True)

        return(df)        
       
    else:
        print("Failed to fetch historical data")
        return None

def update_data(ticker, from_date: dt.date, to_date: dt.date, lower_strike: float=0.0, upper_strike: float=10000.0):
    global open_interest_df
    put_call_ratio = 0
    total_calls = 0
    total_puts = 0
    max_call_interest = 0
    max_interest_value = 0
    max_put_interest = 0 
    last_price = 0.0
    high_price = 0.0
    low_price = 0.0 
    call_data = None
    put_data = None

    try:
        resp = schwab_client.get_option_chain(ticker, contract_type=schwab.client.Client.Options.ContractType.ALL, strategy=schwab.client.Client.Options.Strategy.ANALYTICAL, from_date=from_date, to_date=to_date, include_underlying_quote=False)
        if resp.status_code in {httpx.codes.OK, httpx.codes.CREATED}:
            data = resp.json()        
            call_data = data.get('callExpDateMap', {})
            put_data = data.get('putExpDateMap', {})
            call_open_interest = calculate_open_interest(call_data, 'CALL')
            put_open_interest = calculate_open_interest(put_data, 'PUT')
            
            # total_open_interest = calculate_total_open_interest(call_open_interest, put_open_interest)
            open_interest_data = []

            # Calculate total open interest for calls and puts
            total_calls = 0
            total_puts = 0
            max_call_interest = 0
            max_put_interest = 0       

            # Iterate over call open interest data for total calls and max call interest
            for strike, oi in call_open_interest.items():
                call_oi = oi['CALL']
                total_calls += call_oi  # Sum total calls across all strike prices
                if lower_strike <= strike <= upper_strike:
                    if call_oi > max_call_interest:
                        max_call_interest = call_oi

            # Iterate over put open interest data for total puts and max put interest
            for strike, oi in put_open_interest.items():
                put_oi = oi['PUT']
                total_puts += put_oi  # Sum total puts across all strike prices
                if lower_strike <= strike <= upper_strike:
                    if put_oi > max_put_interest:
                        max_put_interest = put_oi   

            max_interest_value = max(max_call_interest, max_put_interest)

            put_call_ratio = total_puts / total_calls if total_calls != 0 else float('inf')

            for strike in set(call_open_interest.keys()).union(set(put_open_interest.keys())):
                calls_oi = call_open_interest.get(strike, {}).get('CALL', 0)
                puts_oi = put_open_interest.get(strike, {}).get('PUT', 0)
                open_interest_data.append({'Strike': strike, 'Calls': calls_oi, 'Puts': puts_oi})

            open_interest_df = pd.DataFrame(open_interest_data)         
            
        resp = schwab_client.get_quote(ticker)
        if resp.status_code in {httpx.codes.OK, httpx.codes.CREATED}:
            data = resp.json()
            quote = data.get(ticker).get('quote')
            open_price = round(quote.get('openPrice'), 2)
            last_price = round(quote.get('lastPrice'), 2)
            high_price = round(quote.get('highPrice'), 2)
            low_price = round(quote.get('lowPrice'), 2)

        return open_interest_df, open_price, last_price, high_price, low_price, put_call_ratio, max_interest_value  
    except OAuthError as error:
        print()
        print(f'Error: {error}')
        print()
        exit(-1) 

# Function to convert a zero-based index to a column letter (0 -> B, 26 -> AA, etc.)
def index_to_column_letter(index):
    index += 2  # Adjust index to start at 0 for column B
    letter = ""
    
    while index > 0:
        index -= 1  # Adjust for 0-based indexing
        letter = chr(index % 26 + ord('A')) + letter
        index //= 26
        
    return letter
    
def main():
    global schwab_client
    global expiry_date_string
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--reauth', action='store_true', help="Force reauthorize?")    
    parser.add_argument('--expiry', type=str, help="Expiry date")
    parser.add_argument('--add_ticker', type=str, help="Ticker to add to ticker list")
    args = parser.parse_args()

    if args.reauth:
        schwab_client = initialize_schwab_client(reauthorize=True)
    else:
        schwab_client = initialize_schwab_client(reauthorize=False)     

    if args.expiry:
        expiry_date_string = str(args.expiry)
 
    tickers = ["AMD", "SPY", "QQQ", "AAPL", "NVDA", "TQQQ", "UPRO", "NVDL"]

    if args.add_ticker:
        tickers.append(str(args.add_ticker).upper())
        
    ticker = 'upro'.upper()
    expiry_date_string = None
    output_filename = "./maxpain_results.txt"
    is_friday = False

    # hist = get_ohlc_history(ticker, days=5)
    # adr5 = calculate_adr(hist, 5)

    expiry_date_string = get_the_next_friday(dt.datetime.now().date())
    
    from_date_obj = dt.datetime.strptime(expiry_date_string, "%Y-%m-%d").date()
    to_date_obj = dt.datetime.strptime(expiry_date_string, "%Y-%m-%d").date()

    if os.path.exists(output_filename):
        os.remove(output_filename)

    print(f'For expiry date: {expiry_date_string}')
    print(f"Getting Max Pain and Put/Call Ratios for {len(tickers)} symbols...")
    data = [] 
    output = []   

    for ticker in tickers:
        open_interest_df, open_price, last_price, high_price, low_price, put_call_ratio, max_interest_value = update_data(ticker, from_date=from_date_obj, to_date=to_date_obj)
        
        max_pain = calculate_max_pain(open_interest_df)
        
        max_pain_value = f'{max_pain:1.1f}'  
        put_call_ratio_value = f'{put_call_ratio:1.2f} '     

        print('.', end='', flush=True)  
        
        with open(output_filename, 'a') as f:
            f.write(max_pain_value + "\t\t\t")            
        data.append([ticker, max_pain_value, put_call_ratio_value])

        output.append(f'${float(max_pain_value):1.2f}')                

    time.sleep(1)
    print()
    print(tabulate(data, headers=['Ticker', "Maxpain", "Put/Call"], tablefmt="pretty"))  
    print()

    print("Updating Max Pain Spreadsheet...")

    # Define the required scope for Google Sheets
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

    # Load credentials from the downloaded JSON key file
    creds = Credentials.from_service_account_file("./maxpain-436613-be003f817be0.json", scopes=SCOPES)

    # Use gspread to authenticate and open the Google Sheet
    client = gspread.authorize(creds)
    spreadsheet = client.open("Max Pain Daily")

    # Open the sheet by index (0 for first sheet)
    sheet = spreadsheet.get_worksheet(0)

    # Get the current number of rows with data (excluding empty rows)
    current_row_count = len(sheet.get_all_values())    

    # The next available row (1-indexed)
    next_row = current_row_count + 1

    print(f'Appending to make row {next_row}')

    # Create a format object for yellow background
    yellow_background = CellFormat(backgroundColor=color(1, 1, 0))  # RGB for yellow

    # Today's date
    current_date = datetime.now().date()    

    # Format the date without leading zeros for month and day
    formatted_date = f"{current_date.month}/{current_date.day}/{current_date.year}"

    # Data to add as a new row
    new_row = []
    new_row.append(formatted_date)

    # Initialize the starting column (C is column 3, so 'C' = chr(67))
    start_column = ord('C')
    
    # Iterate over your items and increment columns dynamically for the same row
    for index, item in enumerate(output):
        # Add the current item to the new_row
        new_row.append(item)
        
        # Calculate the current column letters for each pair of columns
        first_column = chr(start_column + 3 * index)  # C, F, I, L, etc.        
        second_column = chr(start_column + 3 * index + 1)  # D, G, J, M, etc.            

        previous_column_1 = chr(ord(second_column) - 1)
        previous_column_2 = chr(ord(previous_column_1) - 1)
        
        # Append the dynamic column/row references for the same row
        formula = f'=IF(DATEVALUE($A{next_row}) = TODAY(), GOOGLEFINANCE(INDEX(SPLIT({first_column}$1, " "), 1, 1), "price"), INDEX(GOOGLEFINANCE(INDEX(SPLIT({first_column}$1, " "), 1, 1), "close", $A{next_row}), 2, 2))'
        new_row.append(formula)
        formula = f'={previous_column_1}{next_row}-{previous_column_2}{next_row}'
        new_row.append(formula)
    
    # Append the row
    sheet.append_row(new_row, value_input_option=gspread.utils.ValueInputOption.user_entered)
    
    # Check if Friday
    if current_date.weekday() == 4:
        format_cell_range(sheet, f'A{next_row}:Z{next_row}', yellow_background)

    print("Row added successfully!")  
    print()
    
if __name__=='__main__':
    main()