class Portfolio():
    def __init__(self, balance: int):
        self.stocks = dict() #Contains {ticker: quantity}
        self.balance = balance

    """
    def get_portfolio_value(self):
        value = 0

        for ticker in self.stocks:
            value+=self.stocks[ticker]

        return value
    """
    def sell(self, ticker: str, price: int, quantity: int):
        if ticker not in self.stocks or self.stocks[ticker] < quantity: return -1

        self.stocks[ticker]-=quantity

        self.balance+=quantity*price
        if self.stocks[ticker] == 0: del ticker

        return self.balance
    
    def buy(self, ticker: str, price: int, quantity: int):
        if self.balance < quantity*price: return -1
        if ticker not in self.stocks: self.stocks[ticker] = quantity
        else: self.stocks[ticker]+=quantity

        self.balance-=quantity*price
        return self.balance