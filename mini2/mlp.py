from tqdm import tqdm
import numpy as np
import sklearn as sk
import pandas as pd
import sklearn.model_selection
import matplotlib.pyplot as plt
import random
import seaborn as sns

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


def show_mnist(training_data):
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def show_cifar(training_data):
    labels_map = {
        0: "Airplane",
        1: "Automobile",
        2: "Bird",
        3: "Cat",
        4: "Deer",
        5: "Dog",
        6: "Frog",
        7: "Horse",
        8: "Ship",
        9: "Truck",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        # Change Channels x Height x Width (C, H, W) to Height x Width x Channels (H, W, C)
        plt.imshow(img.permute(1, 2, 0))
    plt.show()


training_data_mnist = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data_mnist = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

training_data_cifar = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data_cifar = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_images_mnist = training_data_mnist.data.numpy()
train_labels_mnist = training_data_mnist.targets.numpy()

test_images_mnist = test_data_mnist.data.numpy()
test_labels_mnist = test_data_mnist.targets.numpy()

train_images_cifar = training_data_cifar.data
train_labels_cifar = np.array(training_data_cifar.targets)

test_images_cifar = test_data_cifar.data
test_labels_cifar = np.array(test_data_cifar.targets)

show_mnist(training_data_mnist)
show_cifar(training_data_cifar)
