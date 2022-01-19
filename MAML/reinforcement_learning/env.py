from dis import dis
import math
import random

import numpy as np
import matplotlib.pyplot as plt


class Navigation2D:
    def __init__(self, board_size=1.0, action_min=-0.1, action_max=0.1):
        self.board_size = board_size
        
        self.action_min = action_min
        self.action_max = action_max

        self.prev_action = np.zeros(2)

    def reset(self, landmark_position):
        # landmark position: 2d ndarray
        self.agent_position = np.array([0.5, 0.5])
        self.landmark_position = landmark_position
        self.env_step = 0
        return self.get_observation()

    def step(self, action):
        # action: 2d ndarray
        action = action.detach().numpy()
        action = np.clip(action, self.action_min, self.action_max)
        self.agent_position = self.agent_position + 0.5*(self.prev_action + action) + np.random.normal(0, 0.1, 2)
        for i in range(2):
            if self.agent_position[i] < 0.0:
                self.agent_position[i] = abs(self.agent_position[i])
            elif self.agent_position[i] > self.board_size:
                self.agent_position[i] = 2*self.board_size - self.agent_position[i]
        self.prev_action = action
        self.env_step += 1

        observation = self.get_observation()
        reward = self.get_reward()
        done = self.get_done(reward)
        return observation, reward, done

    def get_observation(self):
        return self.agent_position
    
    def get_reward(self):
        return -np.sqrt(np.sum((self.agent_position - self.landmark_position)**2))

    def get_done(self, reward):
        distance_x = np.abs(self.agent_position[0] - self.landmark_position[0])
        distance_y = np.abs(self.agent_position[1] - self.landmark_position[1])
        return (distance_x < 0.01 and distance_y < 0.01) or self.env_step >= 100


if __name__ == "__main__":
    # Demo
    env = Navigation2D()
    state = env.reset(landmark_position = np.random.uniform(0.0, 1.0, 2))
    agent_position_list = []
    for i in range(100):
        action = np.random.uniform(-0.1, 0.1, 2)
        state, reward, done = env.step(action)
        print(f'state: {state}\t, reward: {reward}\t, done: {done}')
        agent_position_list.append(env.agent_position.tolist())

    plt.style.use('ggplot')
    x, y = zip(*agent_position_list)
    plt.scatter(x[0], y[0], marker='o', color='mediumpurple')
    plt.plot(x, y, linewidth=1.0, color='mediumpurple')
    plt.scatter(env.landmark_position[0], env.landmark_position[1], marker='x')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.show()