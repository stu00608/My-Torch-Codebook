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

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])

    train_dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data",
        train=False,
        transform=transform
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



