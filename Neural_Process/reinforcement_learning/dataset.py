import torch
from torch.utils.data import Dataset


class CartPoleDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]