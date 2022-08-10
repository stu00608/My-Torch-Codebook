import os
import sys
import yaml
import torch
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from datasets.mnist_dataset import mnist_dataset
from models.mnist_model import MnistAutoencoderModel
from utils.files import file_choices
from utils.helper import set_device, to_numpy, to_tensor

PATHS = yaml.safe_load(open("paths.yaml"))
for k in PATHS:
    sys.path.append(PATHS[k])


class MnistAutoencoderSolver:
    def __init__(self, config, gpu_id) -> None:
        config = yaml.safe_load(open(PATHS["CONFIG"] + config))
        self.__dict__.update({}, **config)
        if self.use_wandb:
            upload_config = {}
            upload_config.update(config["model_params"])
            upload_config.update(config["loader_params"])
            wandb.init(config=upload_config,
                       project="My-PyTorch-Codebook", name=self.run_name)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        set_device(gpu_id)

    def _train(self, dataloader):
        self.model.train()

        # Create a folder to save model weight.
        if not os.path.exists("weights"):
            os.makedirs("weights")

        if self.use_wandb:
            wandb.watch(self.model, log_freq=100)

        for epoch in range(self.model_params["N_epoch"]):

            total_loss = 0.
            for index, (data, label) in enumerate(dataloader):

                # Flatten 28*28 image to 1-D 784 tensor.
                data = data.view(-1, self.model_params["input_size"])

                # Send data and label to compute device.
                data, label = to_tensor(data), to_tensor(label)

                # Reset the gradien inside optimizer.
                self.optimizer.zero_grad()

                # Pass data to model and compute forward.
                reconstructed_data, latent_space = self.model(data)

                # Calcaulate loss between prediction and ground truth.
                loss = self.loss(reconstructed_data, data)

                # Backpropagation
                loss.backward()

                # Gradient step
                self.optimizer.step()

                total_loss += loss

                if index % 10 == 0:
                    print('\nEpoch: {} [{}/{}] \tTraining Loss: {:.3f}'.format(
                        epoch+1,
                        index*len(data),
                        len(dataloader.dataset),
                        loss.item(),
                    ))
                    weight_name = self.run_name + f"_Epoch{(epoch+1):03}.pth"
                    torch.save(self.model, os.path.join("weights", weight_name))

            total_loss /= len(dataloader.dataset)
            if self.use_wandb:
                wandb.log({
                    "train_loss": total_loss,
                })

    def _test(self, dataloader):
        self.model.eval()

        # If there is no weights folder then throw an exception.
        if not os.path.exists("weights"):
            raise RuntimeError(
                "weights folder not exist, please run training first.")

        test_loss = 0.
        with torch.no_grad():

            for i, (data, label) in enumerate(dataloader):

                show_images(data)

                data = data.view(-1, self.model_params["input_size"])

                data, label = to_tensor(data), to_tensor(label)

                reconstructed_data, latent_space = self.model(data)

                loss = self.loss(reconstructed_data, data)

                reconstructed_data = to_numpy(reconstructed_data)
                test_loss += loss

            test_loss /= len(dataloader.dataset)

            print("\nTest Result\nLoss: {:.3f}\n".format(test_loss))

            if self.use_wandb:
                wandb.log({
                    "test Loss": test_loss
                })

    def run(self, is_inference=False):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        train_dataloader, test_dataloader = mnist_dataset(self.loader_params)

        self.model = MnistAutoencoderModel(self.model_params).to(self.device)
        self.loss = nn.MSELoss().to(self.device)
        self.optimizer = Adam(self.model.parameters(),
                              lr=self.model_params["lr"])

        if not is_inference:
            self._train(train_dataloader)
        # self._test(test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=lambda s: file_choices(("yaml"), s), required=True)
    parser.add_argument('--inference', action="store_true")
    parser.add_argument('--gpu_id', type=str, default="")
    args = parser.parse_args()

    solver = MnistAutoencoderSolver(args.config, args.gpu_id)
    solver.run(is_inference=args.inference)
