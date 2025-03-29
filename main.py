print("STARTING")

from env import Env
from model import Actor_Critic
import tensorflow as tf
import numpy as np

print("LOADED PACKAGES")

EPOCHS = 5
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

print("CREATED ENV")

def compute_loss(action_probs, values, returns):
    advantage = returns - values

    log_probs = tf.math.log(action_probs)

    ppo_loss = 0
    value_loss = tf.keras.losses.MSE(values, returns)

    return ppo_loss, value_loss

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


def run_episode(env, agent, steps):
    bal, date = env.reset()
    done = False
    state = env.get_observation()
    total_reward = 0
    for step in range(steps):
        action = agent.take_action(state)

        _, reward, done = env.step(action)
        total_reward+=reward
        if done: break
        state = env.get_observation()

    print(f"Total reward: {total_reward}")

def test(env, agent, epochs, steps):
    for epoch in range(epochs):
        run_episode(env, agent, steps)

actor = Actor_Critic(1, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon, lra=0.9, lrc=0.9)
agent = actor.action_agent
critic = actor.critic

episodes = 5
steps = 1_000

print("BEFORE TRAINING")
test(env, actor, episodes, steps)

for episode in range(episodes):
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

    for step in range(steps):
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
    
    value = critic(state)
    values.append(value)

    states, actions, returns, advantages = preprocess(states, actions, rewards, dones, values, 0.995)

    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch}")
        actor.learn(states, actions, advantages, probs, returns)

print("AFTER TRAINING")
test(env, actor, episodes, steps)