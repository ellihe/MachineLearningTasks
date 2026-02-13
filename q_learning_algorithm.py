# q-learning algorithm for Taxi-v3 environment
# author: EH

import gym
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
import tensorflow as tf
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create environment
env = gym.make("Taxi-v3")
env.reset()
env.render()

action_size = env.action_space.n
print("Action size: ", action_size)

state_size = env.observation_space.n
print("State size: ", state_size)


def eval_policy_better(env_, pi_, gamma_, t_max_, episodes_):
    """
    A function to eval policy
    :param env_: the environment
    :param pi_: the policy list
    :param gamma_: learning rate
    :param t_max_: t max
    :param episodes_: number of episodes
    :return:
    """
    env_.reset()

    v_pi_rep = np.empty(episodes_)
    for e in range(episodes_):
        s_t = env.reset()
        v_pi = 0
        for t in range(t_max_):
            a_t = pi_[s_t]
            s_t, r_t, done, info = env_.step(a_t)
            v_pi += gamma_**t*r_t
            if done:
                break
        v_pi_rep[e] = v_pi
        env.close()
    return np.mean(v_pi_rep), np.min(v_pi_rep), np.max(v_pi_rep), np.std(v_pi_rep)


# Parameters for q-learning algorithm
qtable = np.zeros((500, 6))  # taxi
episodes = 2000  # num of training episodes
interactions = 100  # max num of interactions per episode
epsilon = 0.9  # e-greedy
alpha = 0.5  # learning rate - 1.
gamma = 0.9  # reward decay rate
debug = 0  # for non-slippery case to observe learning
hist = []  # evaluation history

# Main Q-learning loop
for episode in range(episodes):
    print("Episode" + str(episode))

    if episode % 100 == 0:
        print(episode)
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for interact in range(interactions):
        # exploitation vs. exploratin by e-greedy sampling of actions
        if np.random.uniform(0, 1) < epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = np.random.randint(0, 6)

        # Observe
        new_state, reward, done, info = env.step(action)
        new_state = np.reshape(new_state, [1, 1])

        # Update Q-table

        qtable[state, action] = qtable[state, action] + alpha * (
                    reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        # Our new state is state
        state = new_state

        # Check if terminated
        if done:
            break

    if episode % 10 == 0 or episode == 1:
        # print(reward)
        pi = np.argmax(qtable, axis=1)
        val_mean, val_min, val_max, val_std = eval_policy_better(env, pi, gamma, interactions, 1000)
        # val_mean, val_min, val_max, val_std = eval_policy(env, pi, gamma)#, interactions, 1000)
        hist.append([episode, val_mean, val_min, val_max, val_std])
        if debug:
            print(pi)
            print(val_mean)

# Plot things
env.reset()
print(qtable)
hist = np.array(hist)
print(hist.shape)

plt.plot(hist[:, 0], hist[:, 1])
plt.show()

hist = np.array(hist)
print(hist.shape)

plt.plot(hist[:, 0], hist[:, 1])
# Zero-clipped
# plt.fill_between(hist[:,0], np.maximum(hist[:,1]-hist[:,4],np.zeros(hist.shape[0])),hist[:,1]+hist[:,4],
plt.fill_between(hist[:, 0], hist[:, 1]-hist[:, 4], hist[:, 1]+hist[:, 4],
                 alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', linewidth=0)
plt.show()


def one_hot_action(state):
    """
    Function to encode one_hot state
    :param state: current state as a integer
    :return: one_hot vector
    """
    onehot = np.zeros(6)
    hot = np.argmax(qtable[state])
    onehot[hot] = 1
    return onehot


def neural_network(states):
    """
    A function to create, train and test neural network for taxi-v3 environment
    :param states: the sample size
    :return: none
    """
    # Neural network
    model = tf.keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(500,)),
        layers.Dense(10, activation='relu'),
        layers.Dense(6, activation='softmax')
    ])

    model.compile(loss='mse', optimizer='adam')

    one_hot = np.eye(states, 500)
    y = np.zeros((states, 6))

    for state in range(states):
        y[state, :] = one_hot_action(state)

    history = model.fit(one_hot, y, epochs=200, batch_size=32)

    # Plot a loss function
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.set_title('Neural network training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    plt.show()

    # test neural network
    one_hot = np.eye(500)
    action_pred = np.argmax(model.predict(one_hot), axis=1)
    action_true = np.argmax(qtable, axis=1)
    print(action_true == action_pred)
    MSE.append(np.sum(action_true == action_pred))
    '''tests = 1
    env = gym.make("Taxi-v3")

    for j in range(tests):
        state = env.reset()
        one_hot = np.zeros(500)
        one_hot[state] = 1
        state = one_hot
        total_reward = 0

        for step in range(100):
            action = np.argmax(model.predict(np.array([state]))[0])
            state, reward, done, info = env.step(action)
            #next_state = np.reshape(next_state, [1, 1])
            total_reward += reward
            #state = next_state
            one_hot = np.zeros(500)
            one_hot[state] = 1
            state = one_hot

            if done:
                print(step)
                break

        print("Test Total Reward: {}".format(total_reward))'''


MSE = []
statesv = []
# Train neural network for different sample sizes
for states in range(50, 501, 50):
    statesv.append(states)
    neural_network(states)

# Plot interesting things
fig, ax = plt.subplots()
ax.plot(statesv, 500*np.ones(10,)-MSE)
ax.set_title('Model goodness as a function of samples ')
ax.set_xlabel('Number of samples')
ax.set_ylabel('Number of different actions')
plt.show()
