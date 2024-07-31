# Neural network simulation, includes training and testing
# author: Ellinoora Hetemaa

import gym
from keras.models import Sequential
from keras.layers import Embedding, Reshape, Dense
import numpy as np

# create Q-table
num_states = 500
num_actions = 6
alpha = 0.5  # learning rate - 1.
q_table = np.zeros((num_states, num_actions))

env = gym.make("Taxi-v3")
env.reset()
env.render()

# create neural network
model = Sequential()
model.add(Embedding(num_states, 10, input_length=1))
model.add(Reshape((10,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_actions, activation='linear'))
model.compile(loss='mse', optimizer='adam')

# train neural network with Q-table
num_episodes = 100
max_steps = 100
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, 1])
    total_reward = 0
    for step in range(max_steps):
        # choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(state))

        # take action and observe new state and reward
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, 1])

        # update Q-table with new knowledge
        q_target = reward + gamma * np.amax(q_table[next_state[0,0], :])
        q_table[state[0, 0], action] = (1 - alpha) * q_table[state[0,0], action] + alpha * q_target

        # train neural network with Q-table
        q_values = model.predict(state)
        q_values[0, action] = q_table[state[0, 0], action]

        # update total reward and state
        total_reward += reward
        state = next_state

        # check if episode is finished
        if done:
            break

    # update epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # print results
    print("Episode: {}, Steps: {}, Total Reward: {}, Epsilon: {:.4f}".format(episode, step, total_reward, epsilon))

# test neural network
state = env.reset()
state = np.reshape(state, [1, 1])
total_reward = 0
for step in range(max_steps):
    action = np.argmax(model.predict(state))
    next_state, reward, done, info = env.step(action)
    next_state = np.reshape(next_state, [1, 1])
    total_reward += reward
    state = next_state
    if done:
        break

print("Test Total Reward: {}".format(total_reward))
