from env import Env
from model import Actor_Critic
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Convert states, actions, rewards, etc. into data used for PPO algorithm
def preprocess(states, actions, rewards, dones, values, gamma):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * dones[i] - values[i]
        g = delta + gamma * lmbda * dones[i] * g
        returns.append(g + values[i])

    returns.reverse()
    advantages = np.array(returns, dtype=np.float32) - values[:-1]
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    returns = np.array(returns, dtype=np.float32)

    return states, actions, returns, advantages

# Runs a single episode, set output or graph to true to view additional information
def run_episode(env, actor, steps, output=False, graph=False):
    bal, date = env.reset()

    if graph: datapoints = []
    done = False
    state = env.get_observation()
    total_reward = 0

    for step in range(steps):
        action = actor.take_action(state)
        _, reward, done = env.step(action)

        if output:
            print(f"Date: {env.date}, Step: {step}, Action: {action}, Reward: {reward}")

        if graph:
            datapoints.append((env.date, env.portfolio.get_portfolio_value() + env.portfolio.balance))

        total_reward+=reward
        if done: break
        state = env.get_observation()

    if graph:
        plt.plot([i[0] for i in datapoints], [i[1] for i in datapoints])
        plt.show()

    return total_reward

# Runs multiple epochs and logs the reward
def test(env, actor, epochs, steps):
    for epoch in range(epochs):
        print("EPOCH:", epoch+1)
        reward = run_episode(env, actor, steps)
        print(f"Reward for epoch {epoch+1} : {reward} ")

# Init params
EPOCHS = 5
MAX_TIMESTEPS = 100
EPISODES = 10
STEPS = 1000

stocks = [
    "AMZN",
    "GOOGL",
    "EA"
]

epsilon = 0.9
epsilon_decay = 0.95
min_epsilon = 0.05
noise_stdev=5
noise_decay=0.995
min_noise=0.1
gamma = 0.95

balance = 100_000 # 100,000 starting balance
date = "2011-01-03" # Starting date

env = Env(balance, date, stocks)

actor = Actor_Critic(number_of_stocks=len(stocks), epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon, noise_stdev=noise_stdev, noise_decay=noise_decay, min_noise=min_noise, lra=0.01, lrc=0.01)
agent = actor.action_agent
critic = actor.critic

# Test the model before training
"""
test(env, actor, EPISODES, STEPS)
run_episode(env, actor, STEPS, graph=True)
"""

# Training loop
for episode in range(EPISODES):
    actor.set_epsilon(0.9)
    done = False
    bal, date = env.reset()
    state = env.get_observation()
    actor_loss = []
    critic_loss = []
    rewards = []
    states = []
    actions = []
    probs = []
    dones = []
    values = []

    for step in range(STEPS):
        action = actor.take_action(state)
        value = critic(state)

        _, reward, done = env.step(action)
        next_state = env.get_observation()

        dones.append(1-done)
        rewards.append(reward)
        states.append(state)
        actions.append(action)

        prob = agent(state)
        probs.append(prob)
        values.append(value)
        state = next_state

        if done:
            env.reset()
            state = env.get_observation()
        actor.adjust_epsilon()
    
    value = critic(state)
    values.append(value)

    states, actions, returns, advantages = preprocess(states, actions, rewards, dones, values, 0.995)

    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch}")
        actor.learn(states, actions, advantages, probs, returns)

# Test the model after training
"""
test(env, actor, EPISODES, STEPS)
run_episode(env, actor, STEPS, graph=True)
"""