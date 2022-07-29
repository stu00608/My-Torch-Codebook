# Initialize paths
import sys
import os
import yaml
import torch
from torchvision import datasets, transforms

PATHS = yaml.safe_load(open("paths.yaml"))
for k in PATHS:
    sys.path.append(PATHS[k])

def mnist_dataset(params):
    """Generate MNIST Dataloader."""

    if not os.path.exists("./data"):
        os.mkdir("./data")

    train_dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    test_dataset = datasets.MNIST(
        "./data",
        train=False,
        transform=transforms.ToTensor()
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params["batch_size"],
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params["batch_size"],
        shuffle=True
    )

    return train_dataloader, test_dataloader



