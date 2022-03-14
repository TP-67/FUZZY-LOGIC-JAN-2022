import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt


def preprocessing(data: np.array) -> np.array:
    data = data[1:]
    temp_data = np.zeros((data.shape[0], data.shape[1] + 1))

    for i in range(len(data)):
        if data[i][2] == 'Dog':
            data[i][2] = 0
        elif data[i][2] == 'Cat':
            data[i][2] = 1

    data = data.astype(np.float)
    data = torch.tensor(data, dtype=torch.float32)

    # Normalization
    data[:, 0] = (data[:, 0] - torch.min(data[:, 0])) / (torch.max(data[:, 0]) - torch.min(data[:, 0]))
    data[:, 1] = (data[:, 1] - torch.min(data[:, 1])) / (torch.max(data[:, 1]) - torch.min(data[:, 1]))
    # data[:, :2] = F.normalize(data[:, :2], dim=1)

    return data


class Animals(Dataset):
    def __init__(self, root: str, transform=None):
        self.root: str = root
        self.data = preprocessing(np.genfromtxt(self.root, delimiter=',', dtype=str))

        dog_cluster = self.data[(self.data[:, 2] == 0).nonzero().flatten()]
        cat_cluster = self.data[(self.data[:, 2] == 1).nonzero().flatten()]

        # Plots
        plt.figure(0)
        plt.scatter(dog_cluster[:, 0], dog_cluster[:, 1], label='Dog')
        plt.scatter(cat_cluster[:, 0], cat_cluster[:, 1], label='Cat')
        plt.legend()
        plt.show()

    def __getitem__(self, index):
        return self.data[index][:2], self.data[index][2]

    def __len__(self):
        return self.data.shape[0]


def get_dataset(root: str):
    return Animals(root)


def get_dataloader(dataset: Dataset, batch_size: int, mode: str):
    shuffle = True if mode == 'train' else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


if __name__ == '__main__':
    pass
