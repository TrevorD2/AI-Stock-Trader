from env import Env
from model import Top_Level_Agent

MAX_TIMESTEPS = 5000

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

tl_agent = Top_Level_Agent(epsilon, epsilon_decay, min_epsilon)

scores = []
num_timesteps = 0
num_episodes = 0

while num_timesteps <= MAX_TIMESTEPS:
    bal, date = env.reset()
    score = 0
    done = False
    
    while not done and num_timesteps <= MAX_TIMESTEPS:

        for ticker in stocks:
            data = env.get_observation(env.date, ticker)
            if data == -1: continue
            output = tl_agent.take_action(data)
            day, reward, done = env.step(output, ticker, 1)

            if done: break

            score+=reward

        env.end_day()
        num_timesteps+=1
        tl_agent.adjust_epsilon()

    scores.append(score)
    num_episodes+=1

    print(f"Episode: {num_episodes}, score: {score}, epsilon: {tl_agent.epsilon}")