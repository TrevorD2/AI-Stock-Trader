import API
from portfolio import Portfolio

class Env():
    def __init__(self, balance: int, start_date: str): #start_date takes the form yyyy/mm/dd
        self.portfolio = Portfolio(balance)
        self.starting_balance = balance

    def step(self, action_tuple): #action = (action, ticker)
        action, ticker = action_tuple

        if action==1:
            self.portfolio.buy(ticker)

        elif action==2:
            self.portfolio.sell(ticker)

        reward = self.portfolio.get_portfolio_value() + self.portfolio.balance - self.starting_balance
        return 0, reward, self.portfolio.get_portfolio_value() < 0 or self.portfolio.balance < 0 #must return observation, reward, terminated