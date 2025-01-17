import requests

def write(text):
    with open("out.txt", "w") as f:
        f.write(str(text))

def get_ticker(ticker: str) -> dict:
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?token=acf127c746b3b029d5c57d622cf64f85aed047fe", headers=headers)
    json = response.json()
    
    return json[0] #Get last day value

def get_ticker(ticker: str, date: str):
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={date}&endDate={date}&token=acf127c746b3b029d5c57d622cf64f85aed047fe", headers=headers)
    json = response.json()
    print(json)
    return json[0] #Get last day value