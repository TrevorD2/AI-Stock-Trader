import requests
import json as js
import pandas as pd
import math

def get_historic_ticker(ticker: str, start_date: str, end_date: str) -> int:
    headers = {
        'Content-Type': 'application/json'
    }

    # Get response from tiingo API, replace [YOUR_API_TOKEN] with valid API token
    response = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token=[YOUR_API_TOKEN]", headers=headers)
    json = response.json()
    if type(json) != list: return 1
    elif len(json) == 0: return 1

    # Convert json from tiingo API to list of dictionaries
    json = [{(ticker+"_"+key if key!="date" else key):(json[i][key][:10] if key == "date" else float(json[i][key])) for key in list(json[i].keys())} for i in range(len(json))] 
    return json

def compile_dataset(stocks: list[str], start_date, end_date):
    
    data = get_historic_ticker(stocks[0], start_date, end_date)
    df = pd.json_normalize(data)

    for stock in stocks[1:]:
        new_data = get_historic_ticker(stock, start_date, end_date)
        new_df = pd.json_normalize(new_data)

        # Remove date column and concatenate to dataframe
        new_df.drop(new_df.columns[0], axis=1, inplace=True)
        df = pd.concat([df, new_df], axis=1)

    return df

# Read data, if error occurs, regenerate data
def read_data():
    try:
        with open("stock_data.json", "r") as f:
            json = js.load(f)
            df = pd.json_normalize(json)
    except:
        df = compile_dataset()
    return df

# Normalize the data by taking the log base 10 and dividing by 10
def log_data():
    df = read_data()
    df = df.map(lambda x: math.log(x)/10 if (type(x)==float and x > 0) else x) #Take the log of data, divide by 10 to get data to be roughly between 0 - 10
    df.to_json("stock_data.json", orient="records")
    return df

# Run this file to create dataset and save to stock_data.json
if __name__ == "__main__":
    stocks = [
        "AMZN",
        "GOOGL",
        "EA"
    ]

    df = compile_dataset(stocks, "2000-01-01", "2025-01-01")
    df.to_json("stock_data.json", orient="records")
    
    df = log_data()
    
    print(df.head(10))