# Neural Process for 1D Regression
import math
import random

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from regression.dataset import SineDataset
from model.training import calculate_loss
from model.utils import context_target_split
from model.neural_process import NeuralProcess

num_context_min = 3
num_context_max = 50
num_extra_target_min = 0
num_extra_target_max = 50

train_epochs = 10000
num_trials = 20

wandb.init(project="Meta-Learning")

if __name__ == "__main__":
    model = NeuralProcess(x_dim=1, y_dim=1, r_dim=128, z_dim=128, h_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=4e-5)

    # Pretrain Neural Process
    num_context_min = 3
    num_context_max = 50
    num_extra_target_min = 0
    num_extra_target_max = 50
    data_loader = DataLoader(SineDataset(), batch_size=16, shuffle=True)

    for epoch in tqdm(range(train_epochs)):
        epoch_loss = 0.0
        for x, y in data_loader:
            num_context = random.randint(num_context_min, num_context_max)
            num_extra_target = random.randint(num_extra_target_min, num_extra_target_max)
            x_context, y_context, x_target, y_target = context_target_split(x, y, num_context, num_extra_target)

            p_y_pred, q_target, q_context = model(x_context, y_context, x_target, y_target)
            loss = calculate_loss(p_y_pred, y_target, q_target, q_context)
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}\tLoss: {epoch_loss}")
        wandb.log({"Loss": epoch_loss})

    torch.save(model.state_dict(), "./regression/model.pt")
    model.load_state_dict(torch.load("./regression/model.pt"))

    # Adapt to unseen scenario
    new_amplitude = np.random.uniform(-1.0, 1.0)
    new_shift = np.random.uniform(-5.0, 5.0)

    context_x = torch.from_numpy(np.random.uniform(-math.pi, math.pi, 10)).float().unsqueeze(0).unsqueeze(-1)
    context_y = new_amplitude * np.sin(context_x - new_shift)

    target_x = torch.linspace(-math.pi, math.pi, 100).float().unsqueeze(0).unsqueeze(-1)

    plt.plot(target_x.detach().numpy().flatten(), new_amplitude * np.sin(target_x.detach().numpy().flatten() - new_shift), color='firebrick', alpha=0.8, label="Ground Truth")
    plt.scatter(context_x.detach().numpy().flatten(), context_y.detach().numpy().flatten(), marker='o', s=20, color="black")
    for i in range(64):
        target_y_mu, target_y_sigma = model(context_x, context_y, target_x)
        if i == 0:
            plt.plot(target_x.detach().numpy().flatten(), target_y_mu.detach().numpy().flatten(), color='blue', alpha=0.1, label="Prediction")
        else:
            plt.plot(target_x.detach().numpy().flatten(), target_y_mu.detach().numpy().flatten(), color='blue', alpha=0.1)
    plt.legend(loc='upper right')
    plt.savefig("./regression/results/prediction.png")

