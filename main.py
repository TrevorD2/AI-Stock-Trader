from env import Env
from model import Agent
import tensorflow as tf

MAX_TIMESTEPS = 100

stocks = [
    "AMZN"
]

epsilon = 1
epsilon_decay = 1
min_epsilon = 0.05

balance = 100000 # 100,000
date = "2022-01-03" #Starting date

env = Env(balance, date)

tl_agent = Agent(1, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
print("ACTION SPACE: ")
print(tl_agent.action_agent.action_space)
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

            data = env.get_observation()
            action, q = tl_agent(data)
            print(f"Agent taking {action} with quantity {q}")
            if action == len(stocks): break
            day, reward, done = env.step((stocks[action], q))
            print(f"Reward for {day} is {reward}")
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