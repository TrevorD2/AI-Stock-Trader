import tensorflow as tf
from tensorflow.keras import Model  # type: ignore
from tensorflow.keras.layers import Dense, Softmax, LSTM # type: ignore
import numpy as np
import math

class Q_Agent(Model):
    def __init__(self, noise_stdev: float, noise_decay: float, min_noise: float):
        super().__init__()
        self.action_space = (0, 0) #Tuple values correspond to (mu, sigma) for a Gaussian distribution
        self.noise_stdev = noise_stdev
        self.noise_decay = noise_decay
        self.min_noise = min_noise

        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(64, activation="relu")
        self.lstm = LSTM(1)
        self.out = Dense(2, activation="tanh")

    #Takes output vector as input, produces continuous probability distribution (mu, sigma)
    def call(self, x): 
        d1 = self.d1(x)
        d2 = self.d2(d1)
        lstm = self.lstm(d2)
        out = self.out(lstm)
        return out

    #Determines quantity given input observationn
    def take_action(self, x, max_shares: int):
        li = self(x).numpy().tolist()
        mu, sigma = li[0]
        sigma = abs(sigma) #Prevent negative standard deviations
        mu*=max_shares
        sigma*=math.sqrt(max_shares)
        quantity = np.random.normal(mu, sigma) + np.random.normal(0, self.noise_stdev)
        return int(max(quantity, 0))

    def adjust_noise(self):
        if self.noise_stdev >= self.min_noise: 
            self.noise_stdev *= self.noise_decay

class Action_Agent(Model):
    def __init__(self, number_of_stocks: int, epsilon: float, epsilon_decay: float, min_epsilon: float):
        super().__init__()
        self.action_space = [i for i in range(number_of_stocks)] # action_space[i] corresponds to the ith stock
        self.action_space.append(number_of_stocks) # Allow the model to choose when to stop trading via an <END> token
        self.number_of_stocks = number_of_stocks

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(64, activation="relu")
        self.lstm = LSTM(1)
        self.d3 = Dense(number_of_stocks+1)
        self.out = Softmax()

        self.q_agent = Q_Agent(epsilon, epsilon_decay, min_epsilon)

    #Takes output vector as input, produces discrete probability distribution
    def call(self, x): 
        d1 = self.d1(x)
        d2 = self.d2(d1)
        lstm = self.lstm(d2)
        d3 = self.d3(lstm)
        out = self.out(d3)
        return out

    def take_action(self, data):
        if np.random.random() <= self.epsilon: 
            action = np.random.choice(self.action_space)

        else: action = np.argmax(self(data), axis=1)
        return action

    def adjust_epsilon(self):
        if self.epsilon >= self.min_epsilon: 
            self.epsilon *= self.epsilon_decay

class Value_Agent(Model):
    def __init__(self):
        super().__init__()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(64, activation="relu")
        self.lstm = LSTM(1)
        self.out = Dense(1, activation="tanh")

    def call(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        lstm = self.lstm(d2)
        out = self.out(lstm)
        return out

class Agent(Model):
    def __init__(self, number_of_stocks: int, epsilon: float, epsilon_decay: float, min_epsilon: float):
        super().__init__()
        self.q_agent = Q_Agent(1, 0.99, 0.05)
        self.action_agent = Action_Agent(number_of_stocks, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
        self.number_of_stocks = number_of_stocks


    #Takes input tensor of size DxT corresponding to time-series data
    def call(self, x):
        action = self.action_agent.take_action(x)
        if action == self.number_of_stocks: return (action, 0)
        quantity = self.q_agent.take_action(x, 100)

        return (action, quantity)
    
    def adjust_epsilon(self):
        self.q_agent.adjust_noise()
        self.action_agent.adjust_epsilon()

