import requests

headers = {
    'Content-Type': 'application/json'
}

requestResponse = requests.get("https://api.tiingo.com/tiingo/daily/aapl/prices?startDate=2019-01-02&token=acf127c746b3b029d5c57d622cf64f85aed047fe", headers=headers)

def write(text):
    with open("out.txt", "w") as f:
        f.write(str(text))

write(requestResponse.json())
