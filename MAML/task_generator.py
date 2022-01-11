import math
import random

import numpy as np
import matplotlib.pyplot as plt

import torch


class RegressionTaskGenerator:
    def __init__(self, input_min=-5, input_max=5, amplitude_min=0.1, amplitude_max=5.0):
        self.input_min = input_min
        self.input_max = input_max

        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        
    def get_task(self, num_samples=10):
        amplitude = random.uniform(self.amplitude_min, self.amplitude_max)
        phase = random.random() * math.pi

        x = torch.FloatTensor(np.random.uniform(self.input_min, self.input_max, num_samples)).reshape(-1, 1)
        y = amplitude * torch.sin(phase + x)
        return x, y, amplitude, phase


if __name__ == "__main__":
    task_generator = RegressionTaskGenerator()
    x, y, amplitude, phase = task_generator.get_task()
    plt.scatter(x, y, s=20)
    plt.plot(np.arange(-5.0, 5.0, 0.1), np.sin(np.arange(-5.0, 5.0, 0.1) + phase) * amplitude, linestyle='--', color='lightgreen')
    plt.show()

