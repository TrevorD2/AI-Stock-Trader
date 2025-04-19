class Portfolio():
    def __init__(self, balance: int, input_stocks: list[str]):
        self.stocks = dict() #C ontains {ticker: quantity}

        for stock in input_stocks:
            self.stocks[stock] = 0

        self.balance = balance

        self.current_prices = dict()

        # Initial price does not matter, as only initial obvservation will use this data
        # Initial obvservation will have 0 stocks, so price is irrelevant to observation
        for stock in input_stocks:
            self.current_prices[stock] = 0

    def get_portfolio_value(self):
        value = 0

        for ticker in self.stocks:
            value+=self.stocks[ticker] * self.current_prices[ticker]

        return value
    
    def get_portfolio_observation(self):
        data = [self.balance, self.get_portfolio_value()] + list(self.stocks.values())
        return data

    def sell(self, ticker: str, price: float, quantity: int):
        if ticker not in self.stocks or self.stocks[ticker] < quantity: return -1

        self.stocks[ticker]-=quantity

        self.balance+=quantity*price
        if self.stocks[ticker] == 0: del ticker

        return self.balance
    
    def buy(self, ticker: str, price: float, quantity: int):
        if self.balance < quantity*price: return -1
        else: self.stocks[ticker]+=quantity

        self.balance-=quantity*price
        return self.balance
    
    def update_prices(self, ticker: str, value: float):
        self.current_prices[ticker] = value