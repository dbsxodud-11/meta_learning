import math
import random
import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from task_generator import RegressionTaskGenerator

def load_model(source, target) :
    for source_param, target_param in zip(source.parameters(), target.parameters()) :
        target_param.data.copy_(source_param.data)


class MLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, 40),
                                    nn.ReLU(),
                                    nn.Linear(40, 40),
                                    nn.ReLU(),
                                    nn.Linear(40, self.output_dim)
                                   )

    def forward(self, x):
        out = self.model(x)
        return out

    def parameterised(self, x, weights):
        # like forward, but uses ``weights`` instead of ``model.parameters()``
        # it'd be nice if this could be generated automatically for any nn.Module...
        x = F.linear(x, weights[0], weights[1])
        x = F.relu(x)
        x = F.linear(x, weights[2], weights[3])
        x = F.relu(x)
        x = F.linear(x, weights[4], weights[5])
        return x

class MetaLearner(nn.Module):
    def __init__(self, task="Regression", n_gradient_step=1, n_train_step=10000, alpha=0.01, beta=0.001):
        super(MetaLearner, self).__init__()
        self.n_gradient_step = n_gradient_step
        self.n_train_step = n_train_step
        self.alpha = alpha
        self.beta = beta

        if task == "Regression":
            self.task_generator = RegressionTaskGenerator()
            self.loss_fn = nn.MSELoss()

        self.model = MLP() 
        self.weights = list(self.model.parameters())
        self.meta_optimizer = optim.Adam(self.weights, lr=self.beta)

        log_dir = f'./runs/MAML_{task}' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir)
        self.writer_step = 0

    def train_step(self, batch_size=10, k=10):
        meta_train_loss = 0.0
        for i in range(batch_size):
            temp_weights = [w.clone() for w in self.weights]
            x, y, _, _ = self.task_generator.get_task(num_samples=2*k)
            train_x, test_x = x[:k], x[k:]
            train_y, test_y = y[:k], y[k:]

            train_loss = self.loss_fn(train_y, self.model(train_x))
            grad = torch.autograd.grad(train_loss, self.weights)
            temp_weights = [w - self.alpha * g for w, g in zip(self.weights, grad)]

            test_loss = self.loss_fn(test_y, self.model.parameterised(test_x, temp_weights))
            meta_train_loss += test_loss

        self.meta_optimizer.zero_grad()
        meta_train_loss.backward()
        self.meta_optimizer.step()

        self.writer.add_scalar('Loss/meta_train_loss', meta_train_loss / batch_size, self.writer_step)
        self.writer_step += 1

    def train(self):
        for i in range(self.n_train_step):
            self.train_step()

    def test(self):
        x, y, amplitude, phase = self.task_generator.get_task()
        t = torch.FloatTensor(np.arange(-5.0, 5.0, 0.1)).reshape(-1, 1)
        pre_update = self.model(t).detach()

        test_model = MLP()
        load_model(self.model, test_model)
        test_optimizer = optim.Adam(test_model.parameters(), lr=0.01)

        loss = self.loss_fn(y, test_model(x))
        test_optimizer.zero_grad()
        loss.backward()
        test_optimizer.step()
        one_grad_step = test_model(t).detach()

        for i in range(10):
            loss = self.loss_fn(y, test_model(x))
            test_optimizer.zero_grad()
            loss.backward()
            test_optimizer.step()
        ten_grad_step = test_model(t).detach()

        plt.style.use('ggplot')
        plt.scatter(x, y)
        plt.plot(t, torch.sin(t + phase) * amplitude, color='mediumpurple', label='true distribution')
        plt.plot(t, pre_update, color='limegreen', linestyle=':', label='pre update')
        plt.plot(t, one_grad_step, color='forestgreen', linestyle='--', label='1 grad step')
        plt.plot(t, ten_grad_step, color='darkgreen', linestyle='--', label='10 grad step')
        plt.legend()
        plt.savefig('./results/MAML.png')
        plt.close()

        # Compare with Pretrained Model
        pretrained_model = MLP()
        pretrained_optimizer = optim.Adam(pretrained_model.parameters(), lr=0.01)
        for i in range(self.n_train_step):
            train_x, train_y, _, _ = self.task_generator.get_task()
            loss = self.loss_fn(train_y, pretrained_model(train_x))

            pretrained_optimizer.zero_grad()
            loss.backward()
            pretrained_optimizer.step()

        pre_update = pretrained_model(t).detach()

        test_model = MLP()
        load_model(pretrained_model, test_model)
        test_optimizer = optim.Adam(test_model.parameters(), lr=0.01)

        loss = self.loss_fn(y, test_model(x))
        test_optimizer.zero_grad()
        loss.backward()
        test_optimizer.step()
        one_grad_step = test_model(t).detach()

        for i in range(10):
            loss = self.loss_fn(y, test_model(x))
            test_optimizer.zero_grad()
            loss.backward()
            test_optimizer.step()
        ten_grad_step = test_model(t).detach()
        
        
        plt.style.use('ggplot')
        plt.scatter(x, y)
        plt.plot(t, torch.sin(t + phase) * amplitude, color='mediumpurple', label='true distribution')
        plt.plot(t, pre_update, color='lightcoral', linestyle=':', label='pre update')
        plt.plot(t, one_grad_step, color='firebrick', linestyle='--', label='1 grad step')
        plt.plot(t, ten_grad_step, color='darkred', linestyle='--', label='10 grad step')
        plt.legend()
        plt.savefig('./results/pretrained.png')
        plt.close()

class Pretrained(nn.Module):
    def __init__(self, task="Regression", n_step=10000, lr=0.01):
        super(Pretrained, self).__init__()

        self.n_step = n_step
        self.lr = lr
        self.model = MLP()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if task == "Regression":
            self.task_generator = RegressionTaskGenerator()
            self.loss_fn = nn.MSELoss()

    def train(self):
        for i in range(self.n_step):
            x, y, _, _ = self.task_generator.get_task()
            loss = self.loss_fn(y, self.model(x))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    meta_learner = MetaLearner()
    pretrained = Pretrained()
    
    meta_learner.train()
    pretrained.train()

    meta_learner.test()