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
        return quantity

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

        else: action = np.argmax(self(data), axis=1)[0]
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

class Actor_Critic(Model):
    def __init__(self,  number_of_stocks: int, epsilon: float, epsilon_decay: float, min_epsilon: float, lra: float, lrc: float):
        super().__init__()
        self.q_agent = Q_Agent(1, 0.99, 0.05)
        self.action_agent = Action_Agent(number_of_stocks, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
        self.number_of_stocks = number_of_stocks

        self.critic = Value_Agent()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lra)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lrc)
        self.clip_param = 0.2

    #Takes input tensor of size DxT corresponding to time-series data
    def call(self, x):
        action = self.action_agent(x)
        quantity = self.q_agent(x)

        return (action, quantity)
    
    def take_action(self, x):
        action = self.action_agent.take_action(x)
        if action == self.number_of_stocks: return (action, 0)
        quantity = self.q_agent.take_action(x, 100)

        return (action, quantity)
    
    def adjust_epsilon(self):
        self.q_agent.adjust_noise()
        self.action_agent.adjust_epsilon()

    def set_epsilon(self, epsilon):
        self.q_agent.noise_stdev = epsilon
        self.action_agent.epsilon = epsilon


    def learn(self, states, actions, advantages, old_probs, discount_rewards):

        tickers = actions[:, 0]
        tickers = tf.cast(tickers, tf.int32)

        old_probs_tensor = tf.convert_to_tensor(np.array(old_probs), dtype=tf.float32)
        if len(old_probs_tensor.shape) == 3 and old_probs_tensor.shape[1] == 1:
            old_probs_tensor = tf.squeeze(old_probs_tensor, axis=1)

        states_tensor = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
        if len(states_tensor.shape) == 4 and states_tensor.shape[1]==1:
            states_tensor = tf.squeeze(states_tensor, axis=1)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # Forward pass through the action agent to get discrete action probabilities.
            #print("IN GRADIENT TAPE")
            action_probs = self.action_agent(states_tensor, training=True)
            #print(action_probs)
            # For each state in the batch, gather the probability of the taken action.
            batch_indices = tf.range(tf.shape(action_probs)[0])
            indices = tf.stack([batch_indices, tickers], axis=1)
            current_probs = tf.gather_nd(action_probs, indices)
            
            old_current_probs = tf.gather_nd(old_probs_tensor, indices)

            """
            print(type(current_probs), current_probs.shape)
            print(current_probs)
            print(type(old_current_probs))
            print(old_current_probs)
            """

            # Compute the ratio between current and old probabilities.
            ratios = current_probs / (old_current_probs + 1e-10)
            
            # PPO surrogate objective (clipped version).
            surr1 = ratios * advantages
            surr2 = tf.clip_by_value(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # Critic loss: mean-squared error between the discounted rewards and estimated values.
            values = self.critic(states_tensor, training=True)
            critic_loss = 0.5 * tf.reduce_mean(tf.square(discount_rewards - values))
            
        # Get the trainable variables for the actor (here, both the action agent and the q-agent).
        actor_vars = self.action_agent.trainable_variables + self.q_agent.trainable_variables
        
        # Compute gradients.
        actor_grads = tape1.gradient(actor_loss, actor_vars)
        critic_grads = tape2.gradient(critic_loss, self.critic.trainable_variables)
        
        # Apply gradients with the respective optimizers.
        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return actor_loss, critic_loss

"""
    def learn(self, states, actions, advantages, old_probs, discount_rewards):
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            actionprobs = self.action_agent(states, training=True)
            qprobs = self.q_agent(states, training=True)
            vals = self.critic(states, trainign=True)

            diff = tf.math.subtract(discount_rewards, vals)

            critic_loss = 0.5*tf.keras.losses.MSE(discount_rewards, vals)
            actor_loss = self.actor_loss(probs, actions, advantages, old_probs, critic_loss)

        grads1 = tape1.gradient(actor_loss, self.act)
"""
    