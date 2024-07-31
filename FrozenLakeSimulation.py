
#Frozen lake simulation. Calculates statistics first for the deterministic case,
#then for the non-deterministic case with deterministic update rule and
#last for the non-deterministic case with non-deterministic update rule.

#author: Ellinoora Hetemaa


import gym
import numpy as np
import matplotlib.pyplot as plt


def eval_policy(environment, qtable_, num_of_episodes_, max_steps_):
    rewards = []
    for episode in range(num_of_episodes_):
        state = environment.reset()
        step = 0
        done = False
        total_rewards = 0
        for step in range(max_steps_):
            action = np.argmax(qtable_[state, :])
            new_state, reward, done, info = environment.step(action)
            total_rewards += reward

            if done:
                rewards.append(total_rewards)
                break
            state = new_state
    environment.close()
    avg_reward = sum(rewards)/num_of_episodes_
    return avg_reward


def plot_policies(all_episode_vectors, all_reward_vectors, deterministic, axis):
    axis.plot(all_episode_vectors[0], all_reward_vectors[0], 'r', all_episode_vectors[1], all_reward_vectors[1], 'b',
             all_episode_vectors[2], all_reward_vectors[2], 'g', all_episode_vectors[3], all_reward_vectors[3], 'y',
             all_episode_vectors[4], all_reward_vectors[4], 'c', all_episode_vectors[5], all_reward_vectors[6], 'k',
             all_episode_vectors[7], all_reward_vectors[7], 'm', all_episode_vectors[8], all_reward_vectors[8], '--b',
             all_episode_vectors[9], all_reward_vectors[9], '--r')


def calculate_statistics(environment, is_deterministic, isUpdate, axis):
    action_size = environment.action_space.n
    print("Action size: ", action_size)
    state_size = environment.observation_space.n
    print("State size: ", state_size)

    all_episode_vectors = []
    all_reward_vectors = []

    for i in range(0, 10):
        total_episodes = 400
        max_steps = 100
        gamma = 0.9
        alfa = 0.5
        # Random Q-table
        qtable = np.zeros((state_size, action_size))
        rewards = []
        episodes = []
        epsilon = 1
        epsilon_decay = 0.001

        for episode in range(total_episodes):
            state = environment.reset()
            step = 0
            done = False
            reward_tot = 0

            for step in range(max_steps):
                if is_deterministic:
                    rnd = np.random.random()
                    if rnd < epsilon:
                        action = environment.action_space.sample()
                    else:
                        action = np.argmax(qtable[state])
                else:
                    if np.argmax(qtable[state]) == 0:
                        action = environment.action_space.sample()
                    else:
                        action = np.argmax(qtable[state])

                new_state, reward, done, info = environment.step(action)
                if isUpdate:
                    qtable[state, action] += alfa * (reward + gamma*np.argmax([qtable[new_state, :]]) - qtable[state, action])
                else:
                    qtable[state, action] = reward + gamma * np.argmax([qtable[new_state, :]])
                state = new_state
                if done:
                    break

            if episode % 10 == 0:
                rewards.append(eval_policy(environment, qtable, total_episodes, max_steps))
                episodes.append(episode)
                print(f'Best reward after episode {episode} is {eval_policy(environment, qtable, total_episodes, max_steps)}')
            epsilon = max(epsilon - epsilon_decay, 0)
        all_reward_vectors.append(rewards)
        all_episode_vectors.append(episodes)

    plot_policies(all_episode_vectors, all_reward_vectors, is_deterministic, axis)


# Deterministic case
figure, axis = plt.subplots(3, 1)
env1 = gym.make("FrozenLake-v1", is_slippery=False)
env1.reset()
env1.render()
calculate_statistics(env1, False, False, axis[0])

# Non-Deterministic case
env2 = gym.make("FrozenLake-v1", is_slippery=True)
env2.reset()
env2.render()
calculate_statistics(env2, True, False, axis[1])

# Non-Deterministic case with non-deterministic update rule
env3 = gym.make("FrozenLake-v1", is_slippery=True)
env3.reset()
env3.render()
calculate_statistics(env3, True, True, axis[2])


axis[0].set_title("Deterministic FrozenLake")
axis[1].set_title("Non-Deterministic FrozenLake")
axis[2].set_title("Non-Deterministic FrozenLake with update rule")
plt.show()

