from env import Env
from model import Agent
import tensorflow as tf

MAX_TIMESTEPS = 50

stocks = {
    "AMZN",
    "COST",
    "NVDA"
}

epsilon = 0.3
epsilon_decay = 0.99
min_epsilon = 0.05

balance = 100000 # 100,000
date = "2000-01-01" #Starting date
env = Env(balance, date)

tl_agent = Agent()

scores = []
num_timesteps = 0
num_episodes = 0

while num_timesteps <= MAX_TIMESTEPS:
    bal, date = env.reset()
    score = 0
    done = False
    
    while not done and num_timesteps <= MAX_TIMESTEPS:

        for ticker in stocks:
            data = tf.random.uniform(shape=(1, 1, 10, 30))
            #if data == -1: continue
            action, q = tl_agent(data)
            day, reward, done = env.step((action, ticker, q))

            if done: break

            score+=reward

        env.end_day()
        num_timesteps+=1
        tl_agent.adjust_epsilon()

    scores.append(score)
    num_episodes+=1

    print(f"Episode: {num_episodes}, score: {score}, epsilon: {tl_agent.epsilon}")