import requests
import json as js
import pandas as pd
from arch import arch_model
import math

def write(text, file="out", extension=".txt"):
    with open(file+extension, "w") as f:
        f.write(str(text))

def get_historic_ticker(ticker: str, start_date: str, end_date: str) -> int:
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token=7ce3fcc88a7aea6ca85a7cee38bdbffe6e82b3dd", headers=headers)
    json = response.json()
    if type(json) != list: return 1
    elif len(json) == 0: return 1

    json = [{(ticker+"_"+key if key!="date" else key):(json[i][key][:10] if key == "date" else float(json[i][key])) for key in list(json[i].keys())} for i in range(len(json))] 
    return json

def get_ticker(ticker: str) -> dict:
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?token=7ce3fcc88a7aea6ca85a7cee38bdbffe6e82b3dd", headers=headers)
    json = response.json()
    
    return json[0] #Get last day value

def get_ticker(ticker: str, date: str):
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={date}&endDate={date}&token=7ce3fcc88a7aea6ca85a7cee38bdbffe6e82b3dd", headers=headers)
    json = response.json()
    if type(json) != list: return -1
    elif len(json) == 0: return -1

    return json[0] #Get last day value

def compile_dataset(stocks: list[str], start_date, end_date):
    data = []

    for stock in stocks:
        data.extend(get_historic_ticker(stock, start_date, end_date))

    df = pd.json_normalize(data)
    return df

def read_data():
    try:
        with open("stock_data.json", "r") as f:
            json = js.load(f)
            df = pd.json_normalize(json)
    except:
        df = compile_dataset()
    return df

def log_data():
    df = read_data()
    df = df.map(lambda x: math.log(x)/10 if (type(x)==float and x > 0) else x) #Take the log of data, divide by 10 to get data to be roughly between 0 - 10
    df.to_json("stock_data.json", orient="records")
    return df

"""
def fit_GARCH():
    df = read_data()
    model = arch_model(df[])
"""

if __name__ == "__main__":
    stocks = [
        "AMZN",
    ]

    df = compile_dataset(stocks, "2000-01-01", "2025-01-01")
    df.to_json("stock_data.json", orient="records")
    
    df = log_data()
    
    print(df.head(10))
    df = read_data()
    print(df.head(10))