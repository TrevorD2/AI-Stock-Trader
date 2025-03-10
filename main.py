from env import Env
from model import Agent
import tensorflow as tf

MAX_EPISODES = 2
MAX_TIMESTEPS = 100

stocks = [
    "AMZN"
]

epsilon = 1
epsilon_decay = 1
min_epsilon = 0.05
gamma = 0.95

balance = 100000 # 100,000
date = "2022-01-03" #Starting date

env = Env(balance, date, stocks)

tl_agent = Agent(1, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
scores = []
num_episodes = 0


optimizer = tf.keras.optimizers.Adam()

def get_expected_return(rewards, gamma):
    pass

def compute_loss(action_probs, values, returns):
    advantage = returns - values

    log_probs = tf.math.log(action_probs)

    ppo_loss = 0
    value_loss = tf.keras.losses.MSE(values, returns)

    return ppo_loss, value_loss

def train_step():
    action_probs, values, rewards = run_episode()
    returns = get_expected_return(rewards, gamma)

def run_episode():
    rewards = []
    bal, date = env.reset()
    score = 0
    done = False
    num_timesteps = 0
    
    while not done and num_timesteps < MAX_TIMESTEPS:

        action = 0
        while True: # Repeat until agent generates <END> token

            data = env.get_observation()
            action, q = tl_agent(data)
            if action == len(stocks): break

            day, reward, done = env.step((stocks[action], q))
            rewards.append(reward)

            if done: break

            score+=reward
            num_timesteps+=1

        env.end_day()
        tl_agent.adjust_epsilon()

    scores.append(score)

def train_step(self, prediction, label): pass
     

while num_episodes < MAX_EPISODES:
    score = 0
    num_episodes+=1

    print(f"Episode: {num_episodes}, score: {score}, epsilon: {tl_agent.action_agent.epsilon}")
    print(f"Portfolio: {env.portfolio.get_portfolio_value()}, Balance: {env.portfolio.balance}, Stocks: {env.portfolio.stocks}, Prices: {env.portfolio.current_prices}")