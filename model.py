import tensorflow as tf
from tensorflow.keras import Model  # type: ignore
from tensorflow.keras.Layers import Dense, Flatten, Conv2D,Softmax # type: ignore
import env

class Q_Agent(Model):
    def __init__(self):
        self.action_space = (0, 0) #Tuple values correspond to (mu, sigma) for a Gaussian distribution
    

class Top_Level_Agent(Model):
    def __init__(self, stocks: set, env=env.Env):
        super().__init__()
        self.action_space = {0, 1, 2} # 0 = HOLD, 1 = BUY, 2 = SELL
        self.stocks = stocks
        self.env = env

        self.conv1 = Conv2D(filters=16, kernal_size=(3,3), activation="relu")
        self.conv2 = Conv2D(filters=8, kernal_size=(3,3), activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(64, activation="relu")
        self.out = Softmax()

    def call(self, x): #Takes tensor of DxT, corresponding to time-series data
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        flat = self.flatten(conv2)
        d1 = self.d1(flat)
        d2 = self.d2(d1)
        return self.out(d2)