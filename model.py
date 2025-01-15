import tensorflow as tf
from tensorflow.keras import Model  # type: ignore
from portfolio import Portfolio

class Flat_Agent(Model):
    def __init__(self):
        self.action_space = {}
    

class Top_Level_Agent(Model):
    def __init__(self, starting_balance: int):
        self.portfolio = Portfolio(starting_balance)
        self.action_space = {0, 1, 2} # 0 = HOLD, 1 = BUY, 2 = SELL
