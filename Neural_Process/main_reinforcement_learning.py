# Neural Process for Model-Based Reinforcement Learning
import os
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cartpole import CartPoleDataset
from model.training import calculate_loss
from model.utils import context_target_split
from model.neural_process import NeuralProcess
from model.models import Actor, Critic

# Set Hyperparameters
NUMBER_OF_TASKS = 1000
EPOCH_PER_TASK = 256
TRAIN_EPOCHS = 1000
BATCH_SIZE = 16

NUM_CONTEXT_MIN = 3
NUM_CONTEXT_MAX = 128
NUM_EXTRA_TARGET_MIN = 0
NUM_EXTRA_TARGET_MAX = 128

wandb.init(project="Meta-Learning")

# 1. Collect Data for pretraining
env = gym.make("CartPole-v0")
transitions = [[] for _ in range(NUMBER_OF_TASKS)]
for task in tqdm(range(NUMBER_OF_TASKS)):
    # Sample a Task
    env.masscart = np.random.uniform(0.1, 3.0)
    env.polecart = np.random.uniform(0.01, 1.0)

    state = env.reset()
    for _ in range(EPOCH_PER_TASK):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        action_prob = np.zeros(2)
        action_prob[action] = 1.0

        transitions[task].append((state, action_prob, reward, next_state))
        state = next_state

        if done:
            state = env.reset()

for task in range(NUMBER_OF_TASKS):
    state, action, reward, next_state = zip(*transitions[task])
    transitions[task] = (np.stack(state), np.stack(action), np.array(reward).reshape(-1, 1), np.stack(next_state))

dataset = CartPoleDataset(transitions)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 2. Pretrain NP model
model = NeuralProcess(x_dim=6, y_dim=4, r_dim=128, z_dim=128, h_dim=128)
model_optimizer = optim.Adam(model.parameters(), lr=4e-5)


for epoch in tqdm(range(TRAIN_EPOCHS)):
    for state, action, reward, next_state in dataloader:
        
        num_context = random.randint(NUM_CONTEXT_MIN, NUM_CONTEXT_MAX)
        num_extra_target = random.randint(NUM_EXTRA_TARGET_MIN, NUM_EXTRA_TARGET_MAX)

        x = torch.cat([state, action], dim=-1).float()
        y = next_state.float()

        x_context, y_context, x_target, y_target = \
            context_target_split(x, y, num_context, num_extra_target)
        p_y_pred, q_target, q_context = \
            model(x_context, y_context, x_target, y_target)

        model_loss = calculate_loss(p_y_pred, y_target, q_target, q_context)
        model_optimizer.zero_grad()
        model_loss.backward()
        model_optimizer.step()

torch.save(model.state_dict(), "./reinforcement_learning/model.pt")
model.load_state_dict(torch.load("./reinforcement_learning/model.pt"))

# 3. Adapt to unseen task
env.masscart = np.random.uniform(0.1, 3.0)
env.polecart = np.random.uniform(0.01, 1.0)

context = []
reward_list = []

critic = Critic(in_dim=4, out_dim=1)
actor = Actor(in_dim=4, out_dim=2)

critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
actor_optimizer = optim.Adam(actor.parameters(), lr=4e-5)

for i in range(100):
    state = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        if i < 0:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            context.append((state, action, reward, next_state, done))
        else :
            action_prob = actor(torch.from_numpy(state).float().unsqueeze(0))
            action = Categorical(action_prob).sample().item()
            next_state, reward, done, _ = env.step(action)
            context.append((state, action, reward, next_state, done))

            states, actions, rewards, next_states, dones = zip(*random.sample(context, min(len(context), 64)))
            states = torch.from_numpy(np.stack(states)).float()
            actions = torch.from_numpy(np.array(actions).reshape(-1, 1)).float()
            rewards = torch.from_numpy(np.array(rewards).reshape(-1, 1)).float()
            next_states = torch.from_numpy(np.stack(next_states)).float()
            dones = torch.from_numpy(np.array(dones).reshape(-1, 1)).float()

            x = torch.cat([states, actor(states)], dim=-1).unsqueeze(0)
            y = next_states.unsqueeze(0)

            y_pred_mu, y_pred_sigma = model(x, y, x)
            pred_y = Normal(y_pred_mu, y_pred_sigma).rsample()

            pred_values = critic(states)
            target_values = reward + 0.99 * critic(next_states).detach() * (1 - dones)
            critic_loss = F.mse_loss(pred_values, target_values)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss = -critic(y_pred_mu.squeeze(0)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

        state = next_state
        episode_reward += reward

    print(f"Episode: {i+1}\tReward: {episode_reward}")
    wandb.log({"Reward": episode_reward, "Epsiode": i})
    reward_list.append(episode_reward)
wandb.finish()