import tensorflow as tf
from tensorflow.keras import Model  # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D,Softmax # type: ignore
import numpy as np

class Q_Agent(Model):
    def __init__(self):
        self.action_space = (0, 0) #Tuple values correspond to (mu, sigma) for a Gaussian distribution
        

class Top_Level_Agent(Model):
    def __init__(self, epsilon: float, epsilon_decay, min_epsilon):
        super().__init__()
        self.action_space = [0, 1, 2] # 0 = HOLD, 1 = BUY, 2 = SELL
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.conv1 = Conv2D(filters=16, kernel_size=(3,3), activation="relu")
        self.conv2 = Conv2D(filters=8, kernel_size=(3,3), activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(64, activation="relu")
        self.out = Softmax(3)

    def call(self, x): #Takes tensor of DxT, corresponding to time-series data
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        flat = self.flatten(conv2)
        d1 = self.d1(flat)
        d2 = self.d2(d1)
        out = self.out(d2)
        print("OUTPUT OF NETWORK:", out)
        return out

    def take_action(self, data):
        if np.random.random() <= self.epsilon: 
            return np.random.choice(self.action_space)
        
        action = np.argmax(self(data))
        return action

    def adjust_epsilon(self):
        if self.epsilon >= self.min_epsilon: 
            self.epsilon *= self.epsilon_decay

"""
    def forward(self):
        for ticker in self.stocks:
            data = self.env.get_observation(self.env.date, ticker)
            output = self(data)
"""
