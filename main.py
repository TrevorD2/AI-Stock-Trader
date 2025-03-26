from env import Env
from model import Agent, Value_Agent
import tensorflow as tf
import numpy as np

EPOCHS = 2
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
critic = Value_Agent()
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

def ppo_update(actor, critic, optimizer_actor, optimizer_critic, states, actions, log_probs_old, returns, advantages):
    for _ in range(EPOCHS):
        pass

def preprocess(states, actions, rewards, dones, values, gamma):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * done[i] - values[i]
        g = delta + gamma * lmbda * dones[i] * g
        returns.append(g + values[i])

    returns.revserse()
    advantages = np.array(returns, dtype=np.float32) - values[:-1]
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    returns = np.array(returns, dtype=np.float32)

    return states, actions, returns, advantages


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

while num_episodes < EPOCHS:
    score = 0
    num_episodes+=1

    print(f"Episode: {num_episodes}, score: {score}, epsilon: {tl_agent.action_agent.epsilon}")
    print(f"Portfolio: {env.portfolio.get_portfolio_value()}, Balance: {env.portfolio.balance}, Stocks: {env.portfolio.stocks}, Prices: {env.portfolio.current_prices}")

agent = Agent(1, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
critic = Value_Agent()

episodes = 10
steps = 1_000

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
        action = agent.take_action(state)
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

    states, actions, returns, advantage = preprocess(states, actions, rewards, dones, values, 0.995)

    for epochs in range(EPOCHS):
        #TRAIN STEP
        pass


    #TEST