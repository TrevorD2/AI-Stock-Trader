import tensorflow as tf
from tensorflow.keras import Model  # type: ignore

class Q_Agent(Model):
    def __init__(self):
        self.action_space = {}
    

class Top_Level_Agent(Model):
    def __init__(self, starting_balance: int):
        self.action_space = {0, 1, 2} # 0 = HOLD, 1 = BUY, 2 = SELL
        self.proper_time = 0