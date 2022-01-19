import random
import datetime

import numpy as np
from numpy.lib.arraysetops import isin
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

import wandb
wandb.init(project='MAML-RL(Navigation2D)')
from env import Navigation2D

def load_model(source, target) :
    for source_param, target_param in zip(source.parameters(), target.parameters()) :
        target_param.data.copy_(source_param.data)

class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, self.output_dim)
                                   )
        self.init_layers()

    def init_layers(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = torch.FloatTensor(x)
        out = self.model(x)
        return out

    def parameterised(self, x, weights):
        # like forward, but uses ``weights`` instead of ``model.parameters()``
        # it'd be nice if this could be generated automatically for any nn.Module...
        x = torch.FloatTensor(x)
        x = F.linear(x, weights[0], weights[1])
        x = F.relu(x)
        x = F.linear(x, weights[2], weights[3])
        x = F.relu(x)
        x = F.linear(x, weights[4], weights[5])
        return x


class MetaLearner(nn.Module):
    def __init__(self, task="RL", n_gradient_step=1, n_train_step=100, alpha=0.1, beta=0.01, gamma=0.99):
        super(MetaLearner, self).__init__()
        self.n_gradient_step = n_gradient_step
        self.n_train_step = n_train_step
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if task == "RL":
            self.env = Navigation2D()
            self.std = torch.FloatTensor([[0.1, 0], [0, 0.1]])

        self.model = MLP() 
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.beta)

        # log_dir = f'./runs/MAML_{task}_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # self.writer = SummaryWriter(log_dir)
        # self.writer_step = 0

    def train_step(self, batch_size=20, k=20):
        meta_train_loss = 0.0
        for _ in range(batch_size):
            landmark_position = np.random.uniform(0.0, self.env.board_size, 2)

            train_loss = 0.0
            for _ in range(k):
                obs_list = []
                action_list = []
                reward_list = []

                obs = self.env.reset(landmark_position)
                done = False
                while not done:
                    probs = self.model(obs)
                    m = MultivariateNormal(probs, self.std)
                    action = m.sample()
                    next_obs, reward, done = self.env.step(action)
                    
                    obs_list.append(obs)
                    action_list.append(action)
                    reward_list.append(reward)

                    obs = next_obs
                    if done:
                        break
 
                R = torch.tensor([np.sum(reward_list[i:]*(self.gamma**np.arange(i, len(reward_list)))) for i in range(len(reward_list))])
                obs_list = torch.FloatTensor(np.stack(obs_list))
                action_list = torch.FloatTensor(np.stack(action_list))

                probs = self.model(obs_list)
                m = MultivariateNormal(probs, self.std)
                train_loss += torch.sum(-m.log_prob(action_list) * R)

            train_loss /= k
            grad = torch.autograd.grad(train_loss, list(self.model.parameters()))
            temp_weights = [w - self.alpha * g for w, g in zip(list(self.model.parameters()), grad)]

            obs_list = []
            action_list = []
            reward_list = []

            obs = self.env.reset(landmark_position)
            done = False
            while not done:
                probs = self.model.parameterised(obs, temp_weights)
                m = MultivariateNormal(probs, self.std)
                action = m.sample()
                next_obs, reward, done = self.env.step(action)

                obs_list.append(obs)
                action_list.append(action)
                reward_list.append(reward)

                obs = next_obs
                if done:
                    break

            R = torch.tensor([np.sum(reward_list[i:]*(self.gamma**np.arange(i, len(reward_list)))) for i in range(len(reward_list))])
            obs_list = torch.FloatTensor(np.stack(obs_list))
            action_list = torch.FloatTensor(np.stack(action_list))

            probs = self.model.parameterised(obs_list, temp_weights)
            m = MultivariateNormal(probs, self.std)
            meta_train_loss += torch.sum(-m.log_prob(action_list) * R)

        self.meta_optimizer.zero_grad()
        meta_train_loss.backward()
        self.meta_optimizer.step()

        # self.writer.add_scalar('Loss/meta_train_loss', meta_train_loss / batch_size, self.writer_step)
        # self.writer_step += 1
        wandb.log({"Loss/meta_train_loss": meta_train_loss / batch_size})

    def train(self):
        for i in tqdm(range(self.n_train_step)):
            self.train_step()

    def test(self):
        landmark_position = np.array([0.2, 0.8])
        obs = self.env.reset(landmark_position)
        done = False
        pre_update = [obs]
        while not done:
            probs = self.model(obs)
            m = MultivariateNormal(probs, self.std)
            action = m.sample()
            next_obs, reward, done = self.env.step(action)
            obs = next_obs
            pre_update.append(obs)

            if done:
                break

        test_model = MLP()
        load_model(self.model, test_model)
        test_optimizer = optim.Adam(test_model.parameters(), lr=0.01)
        for i in range(3):
            loss = 0.0
            for i in range(40):
                obs_list = []
                action_list = []
                reward_list = []

                obs = self.env.reset(landmark_position)
                done = False
                while not done:
                    probs = test_model(obs)
                    m = MultivariateNormal(probs, self.std)
                    action = m.sample()
                    next_obs, reward, done = self.env.step(action)

                    obs_list.append(obs)
                    action_list.append(action)
                    reward_list.append(reward)

                    obs = next_obs
                    if done:
                        break

                R = torch.tensor([np.sum(reward_list[i:]*(self.gamma**np.arange(i, len(reward_list)))) for i in range(len(reward_list))])
                obs_list = torch.FloatTensor(np.stack(obs_list))
                action_list = torch.FloatTensor(np.stack(action_list))

                probs = test_model(obs_list)
                m = MultivariateNormal(probs, self.std)
                loss += torch.sum(-m.log_prob(action_list) * R)
            
            loss /= 40
            test_optimizer.zero_grad()
            loss.backward()
            test_optimizer.step()

        obs = self.env.reset(landmark_position)
        done = False
        three_grad_step = [obs.tolist()]
        while not done:
            probs = test_model(obs)
            m = MultivariateNormal(probs, self.std)
            action = m.sample()
            next_obs, reward, done = self.env.step(action)

            obs = next_obs
            three_grad_step.append(obs.tolist())
            if done:
                break       

        pre_update_x, pre_update_y = zip(*pre_update)
        plt.plot(pre_update_x, pre_update_y, linewidth=1.0, color='skyblue', linestyle='--', label='pre update')
        three_grad_step_x, three_grad_step_y = zip(*three_grad_step)
        plt.plot(three_grad_step_x, three_grad_step_y, linewidth=1.5, color='navy', label='3 Steps')
        plt.scatter(landmark_position[0], landmark_position[1], marker='x', s=15, color='darkred')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.savefig('./results/MAML.png')
        plt.close()

        # pretrained_model = MLP()
        # pretrained_optimizer = optim.Adam(pretrained_model.parameters(), lr=0.01)

        # for i in tqdm(range(self.n_train_step)):
        #     for i in range(20):
        #         loss = 0.0
        #         pretrained_landmark_position = np.random.uniform(0.0, self.env.board_size, 2)
        #         for _ in range(20):
        #             obs_list = []
        #             action_list = []
        #             reward_list = []

        #             obs = self.env.reset(pretrained_landmark_position)
        #             done = False
        #             while not done:
        #                 probs = pretrained_model(obs)
        #                 m = MultivariateNormal(probs, self.std)
        #                 action = m.sample()
        #                 next_obs, reward, done = self.env.step(action)

        #                 obs_list.append(obs)
        #                 action_list.append(action)
        #                 reward_list.append(reward)

        #                 obs = next_obs
        #                 if done:
        #                     break

        #             R = torch.tensor([np.sum(reward_list[i:]*(self.gamma**np.arange(i, len(reward_list)))) for i in range(len(reward_list))])
        #             obs_list = torch.FloatTensor(np.stack(obs_list))
        #             action_list = torch.FloatTensor(np.stack(action_list))

        #             probs = pretrained_model(obs_list)
        #             m = MultivariateNormal(probs, self.std)
        #             loss += torch.sum(-m.log_prob(action_list) * R)
                
        #         loss /= 20
        #         pretrained_optimizer.zero_grad()
        #         loss.backward()
        #         pretrained_optimizer.step()

        # obs = self.env.reset(landmark_position)
        # done = False
        # pre_update = [obs.tolist()]
        # while not done:
        #     probs = pretrained_model(obs)
        #     m = MultivariateNormal(probs, self.std)
        #     action = m.sample()
        #     next_obs, reward, done = self.env.step(action)

        #     obs = next_obs
        #     pre_update.append(obs.tolist())

        #     if done:
        #         break

        # test_model = MLP()
        # load_model(pretrained_model, test_model)
        # test_optimizer = optim.Adam(test_model.parameters(), lr=0.01)
        # for i in range(3):
        #     loss = 0.0
        #     for i in range(40):
        #         obs_list = []
        #         action_list = []
        #         reward_list = []

        #         obs = self.env.reset(landmark_position)
        #         done = False
        #         while not done:
        #             probs = test_model(obs)
        #             m = MultivariateNormal(probs, self.std)
        #             action = m.sample()
        #             next_obs, reward, done = self.env.step(action)

        #             obs_list.append(obs)
        #             action_list.append(action)
        #             reward_list.append(reward)

        #             obs = next_obs
        #             if done:
        #                 break

        #         R = torch.tensor([np.sum(reward_list[i:]*(self.gamma**np.arange(i, len(reward_list)))) for i in range(len(reward_list))])
        #         obs_list = torch.FloatTensor(np.stack(obs_list))
        #         action_list = torch.FloatTensor(np.stack(action_list))

        #         probs = test_model(obs_list)
        #         m = MultivariateNormal(probs, self.std)
        #         loss += torch.sum(-m.log_prob(action_list) * R)
            
        #     loss /= 40
        #     test_optimizer.zero_grad()
        #     loss.backward()
        #     test_optimizer.step()

        # obs = self.env.reset(landmark_position)
        # done = False
        # three_grad_step = [obs.tolist()]
        # while not done:
        #     probs = test_model(obs)
        #     m = MultivariateNormal(probs, self.std)
        #     action = m.sample()
        #     next_obs, reward, done = self.env.step(action)

        #     obs = next_obs
        #     three_grad_step.append(obs.tolist())

        #     if done:
        #         break     
        
        # pre_update_x, pre_update_y = zip(*pre_update)
        # plt.plot(pre_update_x, pre_update_y, linewidth=1.0, color='limegreen', linestyle='--', label='pre update')
        # three_grad_step_x, three_grad_step_y = zip(*three_grad_step)
        # plt.plot(three_grad_step_x, three_grad_step_y, linewidth=1.5, color='darkgreen', label='3 Steps')
        # plt.scatter(landmark_position[0], landmark_position[1], marker='x', s=15, color='darkred')
        # plt.xlim(0.0, 1.0)
        # plt.ylim(0.0, 1.0)
        # plt.legend()
        # plt.savefig('./results/pretrained.png')
        # plt.close()
        

if __name__ == "__main__":
    meta_learner = MetaLearner()
    
    meta_learner.train()
    meta_learner.test()        

