import API
import pandas_market_calendars as mcal
import tensorflow as tf
import datetime
from portfolio import Portfolio
import pandas as pd
import math
import numpy as np

class Env():
    def __init__(self, balance: int, start_date: str): #start_date takes the form yyyy-mm-dd
        self.init_params = [balance, start_date]
        self.df = API.read_data()
        self.reset()
        
    def reset(self):
        balance, start_date = self.init_params
        self.portfolio = Portfolio(balance)
        self.starting_balance = balance
        self.date = start_date
        return balance, start_date
    
    def step(self, action_tuple): #action = (ticker, quantity)
        ticker, quantity = action_tuple
        action = 1 if quantity>0 else 2

        price = self._get_price(ticker)
        print(f"Price of {ticker} at {self.date} = {price}")
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

    def get_observation(self, return_type="numpy"): # Get observation data for current date. PRECONDITION: date satisfies is_valid_date() and self.date exists
        observation = (self.df.loc[self.df["date"]==self.date])
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
