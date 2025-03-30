import API
import pandas_market_calendars as mcal
import tensorflow as tf
import datetime
from portfolio import Portfolio
import pandas as pd
import math
import numpy as np

class Env():
    def __init__(self, balance: int, start_date: str, stocks: list[str]): #start_date takes the form yyyy-mm-dd
        self.init_params = [balance, start_date, stocks]
        self.df = API.read_data()
        self.reset()
        
    def reset(self):
        balance, start_date, stocks = self.init_params
        self.portfolio = Portfolio(balance, stocks)
        self.starting_balance = balance
        self.last_balance = balance
        self.date = start_date
        self.stocks = stocks
        return balance, start_date
    
    def step(self, action_tuple): #action = (ticker, quantity)
        ticker, quantity = action_tuple
        ticker = int(ticker)
        if quantity == 0: action = -1
        elif quantity > 0: action = 1
        else: action = 2
        
        ticker = self.stocks[ticker] if ticker < len(self.stocks) else -1

        if ticker!=-1:
            price = self._get_price(ticker)
            self.portfolio.update_prices(ticker, price)

        error = 0
        if action==1:
            error = self.portfolio.buy(ticker, price, quantity)

        elif action==2:
            error = self.portfolio.sell(ticker, price, quantity)

        current_total = self.portfolio.get_portfolio_value() + self.portfolio.balance
        if current_total == self.last_balance: reward = 0
        else: reward =  math.tanh(current_total-self.last_balance) if error!=-1 else error

        self.last_balance = current_total

        return self.date, reward, self.portfolio.balance < 0 #must return observation, reward, terminated

    def end_day(self):
        next_date = self.get_date_after_t(self.date, 1)
        self.date = next_date

    def get_observation(self, return_type="numpy"): # Get observation data for current date. PRECONDITION: date satisfies is_valid_date() and self.date exists
        observation = self.df.loc[self.df["date"]==self.date].copy()

        portfolio_observation = self.portfolio.get_portfolio_observation()
        keys = ["balance", "value"] + [ticker for ticker in self.init_params[2]]

        for key in range(len(keys)):
            value = portfolio_observation[key]
            observation.loc[:, keys[key]] = self.normalization(value) if value >0 else 0

        array = observation.iloc[:, 1:]
        if return_type == "numpy":
            array = np.expand_dims(array, axis=0)
        elif return_type == "pandas": pass
        else: raise Exception("Invalid return type given: type must be \"numpy\" or \"pandas\"")

        return array


    #Gets the real cost of ticker for current date
    def _get_price(self, ticker: str):
        data = self.get_observation(return_type="pandas")
        base_price = data[ticker+"_adjOpen"].iloc[0]
        return self._inverse_normalization(base_price)
    
    def normalization(self, price):
        return math.log(price) / 10

    #Reverses log norm (performs e^(10x))
    def _inverse_normalization(self, log_price):
        return math.exp(log_price*10)
    
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
    pass
