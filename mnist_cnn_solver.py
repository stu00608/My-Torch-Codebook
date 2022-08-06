import os
import sys
import yaml
import torch
import wandb
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from datasets.mnist_dataset import mnist_dataset
from models.mnist_cnn_model import MnistCnnModel
from utils.files import file_choices
from utils.helper import set_device, to_numpy, to_tensor

PATHS = yaml.safe_load(open("paths.yaml"))
for k in PATHS:
    sys.path.append(PATHS[k])

class MnistCnnSolver:
    def __init__(self, config, gpu_id) -> None:
        config = yaml.safe_load(open(PATHS["CONFIG"] + config))
        self.__dict__.update({}, **config)
        if self.use_wandb:
            upload_config = {}
            upload_config.update(config["model_params"])
            upload_config.update(config["loader_params"])
            wandb.init(config=upload_config, project="My-PyTorch-Codebook", name=self.run_name)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        set_device(gpu_id)
    
    def _train(self, dataloader):
        self.model.train()
        
        if self.use_wandb:
            wandb.watch(self.model, log_freq=100)

        for epoch in range(self.model_params["N_epoch"]):

            for index, (data, label) in enumerate(dataloader):
                # Send data and label to compute device.
                data, label = to_tensor(data), to_tensor(label)

                # Reset the gradien inside optimizer.
                self.optimizer.zero_grad()

                # Pass data to model and compute forward.
                pred = self.model(data)

                # Calcaulate loss between prediction and ground truth.
                loss = F.nll_loss(pred, label)

                # Backpropagation
                loss.backward()

                # Gradient step
                self.optimizer.step()

                if index % 10 == 0:
                    print('\nEpoch: {} [{}/{}] \tTraining Loss: {:.3f}'.format(
                        epoch+1,
                        index*len(data),
                        len(dataloader.dataset),
                        loss.item(),
                    ))
            if self.use_wandb:
                wandb.log({
                    "train_loss": loss,
                })
    
    def _test(self, dataloader):
        self.model.eval()

        test_loss = 0.
        correct = 0.
        with torch.no_grad():
            for data, label in dataloader:
                data, label = to_tensor(data), to_tensor(label)

                pred = self.model(data)

                test_loss += F.nll_loss(pred, label, reduction="sum").item()
                pred_label = pred.argmax(dim=1, keepdim=True)
                correct += pred_label.eq(label.view_as(pred_label)).sum().item()

            test_loss /= len(dataloader.dataset)
            test_acc = 100. * correct / len(dataloader.dataset)

            print("\nTest Result\nAcc : {:.3f}%\nLoss: {:.3f}\n".format(test_acc, test_loss))

            if self.use_wandb:
                wandb.log({
                    "Test Accuracy": test_acc/100,
                    "test Loss": test_loss
                })

    def run(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        train_dataloader, test_dataloader = mnist_dataset(self.loader_params)

        self.model = MnistCnnModel(self.model_params).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.model_params["lr"])

        self._train(train_dataloader)
        self._test(test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=lambda s:file_choices(("yaml"),s), required=True)        
    parser.add_argument('--gpu_id', type=str, default="")
    args = parser.parse_args()

    solver = MnistCnnSolver(args.config, args.gpu_id)
    solver.run()