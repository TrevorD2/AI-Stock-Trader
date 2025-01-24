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
    if type(json) != list: return -1
    elif len(json) == 0: return -1

    print(f"JSON for {ticker}: ", json)
    return json[0] #Get last day value