import API
import pandas_market_calendars as mcal
import datetime
from portfolio import Portfolio

class Env():
    def __init__(self, balance: int, start_date: str): #start_date takes the form yyyy-mm-dd
        self.init_params = [balance, start_date]
        
    def reset(self):
        balance, start_date = self.init_params
        self.portfolio = Portfolio(balance)
        self.starting_balance = balance
        self.date = start_date
        return balance, start_date
    
    def step(self, action_tuple): #action = (action, ticker, quantity)
        action, ticker, quantity = action_tuple

        price = self._get_price(self.date, ticker)
        self.portfolio.update_prices(ticker, price)

        error = 0
        if action==1:
            error = self.portfolio.buy(ticker, price, quantity)

        elif action==2:
            error = self.portfolio.sell(ticker, price, quantity)

        reward =  self.portfolio.get_portfolio_value() + self.portfolio.balance - self.starting_balance if error != -1 else error

        return self.date, reward, self.portfolio.balance < 0 #must return observation, reward, terminated

    def end_day(self):
        next_date = self.get_date_after_t(self.date, 1)
        self.date = next_date

    def get_observation(self, date: str, ticker: str): # Get observation data for that date. PRECONDITION: date satisfies is_valid_date()
        data = API.get_ticker(ticker, date)
        return data

    def _get_price(self, date: str, ticker: str):
        return self.get_observation(date, ticker)["open"]
    
    #Returns datetime object for closest valid date after or equal to date + t
    def get_date_after_t(self, date: str, t: int):
        yr, m, d = map(int, date.split("-"))
        start_date = datetime.date(yr, m, d)
        end_date = start_date + datetime.timedelta(days=t)

        while(not self.is_valid_date(end_date.strftime("%Y-%m-%d"))):
            end_date+=datetime.timedelta(days=1)

        return end_date.strftime("%Y-%m-%d")

    def is_valid_date(self, date: str):
        nyse = mcal.get_calendar("NYSE")
        return date in nyse.valid_days(start_date=date, end_date=date)
    
if __name__ == "__main__":
    date = "2016-12-23"
    env = Env(10000, date)

    print(env.step((1, "COST", 1)))

    print(env.portfolio.balance, env.portfolio.stocks)

    print(env.step((1, "AMZN", 2)))
    
    print(env.portfolio.balance, env.portfolio.stocks)

    print(env.step((1, "COST", 1)))

    print(env.portfolio.balance, env.portfolio.stocks)

    print(env.step((1, "AMZN", 2)))
    
    print(env.portfolio.balance, env.portfolio.stocks)