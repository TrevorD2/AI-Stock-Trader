from env import Env
from model import Agent
import tensorflow as tf

MAX_TIMESTEPS = 1000

stocks = [
    "AMZN"
]

epsilon = 0.3
epsilon_decay = 0.99
min_epsilon = 0.05

balance = 100000 # 100,000
date = "2001-01-03" #Starting date

env = Env(balance, date)

tl_agent = Agent(1)

scores = []
num_timesteps = 0
num_episodes = 0

while num_timesteps <= MAX_TIMESTEPS:
    bal, date = env.reset()
    score = 0
    done = False
    
    while not done and num_timesteps <= MAX_TIMESTEPS:

        action = 0
        timst = 0
        while timst < 10: # Repeat until agent generates <END> token

            data = env.get_observation().to_numpy()
            action, q = tl_agent(data)
            print(f"AGENT TAKING {action} for {q}")
            if action == -1: break
            else: action = action[0]
            day, reward, done = env.step((stocks[action], q))

            if done: break
            timst+=1
            score+=reward
            num_timesteps+=1

            print("ENDED TIMESTEP:", num_timesteps, env.date, score)
            print(f"Portfolio: {env.portfolio.get_portfolio_value()}, Balance: {env.portfolio.balance}, Stocks: {env.portfolio.stocks}, Prices: {env.portfolio.current_prices}")

        env.end_day()
        tl_agent.adjust_epsilon()
        print("ENDED DAY:", num_timesteps, env.date, score)
        print(f"Portfolio: {env.portfolio.get_portfolio_value()}, Balance: {env.portfolio.balance}, Stocks: {env.portfolio.stocks}, Prices: {env.portfolio.current_prices}")

    scores.append(score)
    num_episodes+=1

    print(f"Episode: {num_episodes}, score: {score}, epsilon: {tl_agent.action_agent.epsilon}")
    print(f"Portfolio: {env.portfolio.get_portfolio_value()}, Balance: {env.portfolio.balance}, Stocks: {env.portfolio.stocks}, Prices: {env.portfolio.current_prices}")