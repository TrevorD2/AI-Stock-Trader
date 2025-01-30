import requests

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

    json = {json[i]["date"][:10]: dict({key: json[i][key] for key in list(json[i].keys())[1:]}) for i in range(len(json))} 
    strjson = str(json).replace("\'", "\"")
    

    write(strjson, f"{ticker}_stock", extension=".json")
    return 0

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

    #print(f"JSON for {ticker}: ", json)
    return json[0] #Get last day value


if __name__ == "__main__":
    get_historic_ticker("AMZN", "2000-01-01", "2025-01-01")