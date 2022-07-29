import wandb
import torch
import torch.nn as nn
import yaml
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from torch.optim import Adam
from models.mnist_model import MnistMlpModel
from utils.helper import set_device, to_numpy, to_tensor
from utils.files import file_choices
from datasets.mnist_dataset import mnist_dataset

PATHS = yaml.safe_load(open("paths.yaml"))
for k in PATHS:
    sys.path.append(PATHS[k])


class MnistSolver:
    """This solver runs a MLP model using MNIST dataset.

    Methods
    -------
    run(gpu_id='')
        Run the whole training process using specific gpu or cpu if blank string was given.
    """

    def __init__(self, config) -> None:
        config = yaml.safe_load(open(PATHS["CONFIG"] + config))
        self.__dict__.update({}, **config)
        if self.use_wandb:
            upload_config = {}
            upload_config.update(config["model_params"])
            upload_config.update(config["loader_params"])
            wandb.init(config=upload_config, project="My-PyTorch-Codebook", name=self.run_name)

        self.progress_folder = os.path.join(PATHS["PROGRESS"], self.run_name)
        if os.path.exists(self.progress_folder):
            files = glob(os.path.join(self.progress_folder, "*.png"))
            for file in files:
                os.remove(file)
        else:
            os.makedirs(self.progress_folder)
        
        self.figure = plt.figure(figsize=(10, 10))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def _train(self, train_dataloader, test_dataloader):
        """Training the model"""
        # Set the model to training mode inplace.

        # Wandb magic.
        if self.use_wandb:
            wandb.watch(self.model, log_freq=100)
        
        total_epoch = self.model_params["N_epoch"]
        for epoch in range(total_epoch):
            print("Train : ", end='')
            self.model.train()
            epoch_loss = 0.
            for index, (r_data, r_label) in enumerate(train_dataloader):
                # Send data and label to compute device.
                data, label = to_tensor(r_data), to_tensor(r_label)

                # Reset the gradien inside optimizer.
                self.optimizer.zero_grad()

                # Pass data to model and compute forward.
                pred_label = self.model(data)

                # Calcaulate loss between prediction and ground truth.
                loss = self.loss_fn(pred_label, label)

                # Backpropagation
                loss.backward()

                # Gradient step
                self.optimizer.step()

                # Summarize loss and acc in every step.
                epoch_loss += to_numpy(loss)

                if index % (len(train_dataloader)//10) == 0:
                    print('.', end='')

            print("\nVal   : ", end='')
            epoch_loss = 100. * epoch_loss / len(train_dataloader.dataset)
            self.model.eval()
            eval_loss = 0.0
            for index, (r_data, r_label) in enumerate(test_dataloader):
                data, label = to_tensor(r_data), to_tensor(r_label)
                pred_label = self.model(data)
                loss = self.loss_fn(pred_label, label)
                eval_loss += to_numpy(loss)

                if index % (len(test_dataloader)//10) == 0:
                    print('.', end='')

            eval_loss = 100. * eval_loss / len(test_dataloader.dataset)
            
            print('\nEpoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch+1, 
                total_epoch,
                epoch_loss,
                eval_loss
            ))

            if self.use_wandb:
                wandb.log({
                    "train_loss": epoch_loss,
                    "test_loss": eval_loss
                })
                

    def _predict(self, dataloader):
        # Change model to evaluation mode inplace.
        # NOTE: WIP, need to return the label not prob.
        self.model.eval()
        with torch.no_grad():
            for r_data, r_label in dataloader:
                data, label = to_tensor(r_data), to_tensor(r_label)
                pred_label = self.model(to_tensor(data))

        return pred_label

    def _log_progress(self, epoch, data, label):
        # NOTE: Circle data version, not yet implement into mnist task.
        plt.clf()
        data_x, data_y, label = to_numpy(
            data[:, 0]), to_numpy(data[:, 1]), to_numpy(label)
        plt.subplot(111)
        plt.ylabel("y")
        plt.xlabel("x")
        plt.scatter(data_x, data_y, c=label)

        plt.savefig(os.path.join(self.progress_folder, "epoch_%03d.png" % epoch))

    def run(self, gpu_id):
        set_device(gpu_id)

        # Set seeds manually, so the result is reproducable.
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Get train and test dataset and fit into dataloader.
        train_dataloader, test_dataloader = mnist_dataset(self.loader_params)

        # Create model object and loss function and optimizer function.
        self.model = MnistMlpModel(self.model_params).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.model_params["lr"])
        self.loss_fn = nn.CrossEntropyLoss()

        # Train
        self._train(train_dataloader, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=lambda s:file_choices(("yaml"),s), required=True)        
    parser.add_argument('--gpu_id', type=str, default="")
    args = parser.parse_args()

    solver = MnistSolver(args.config)
    solver.run(args.gpu_id)
