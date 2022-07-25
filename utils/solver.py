import wandb
import torch
import torch.nn as nn
import yaml
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.circle import CircleRegressor
from utils.helper import set_device, to_numpy, to_tensor
from utils.dataset import CircleData
from utils.metrics import mse

PATHS = yaml.safe_load(open("paths.yaml"))
for k in PATHS: sys.path.append(PATHS[k])

class CircleSolver:
    """This solver runs a regression training model that predict the label of inner or outer circle.
    
    Methods
    -------
    run(gpu_id='')
        Run the whole training process using specific gpu or cpu if blank string was given.
    """

    def __init__(self, config) -> None:
        config = yaml.safe_load(open(PATHS["CONFIG"] + config))  
        self.__dict__.update({}, **config)
        if self.use_wandb:
            wandb.init(config=config, project="My-PyTorch-Codebook")

        self.progress_data_x = []
        self.progress_data_y = []
        self.progress_data_label = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _train(self, dataloader):
        """Training the model"""
        # Set the model to training mode inplace.
        self.model.train()
        total_epoch = self.model_params["N_epoch"]

        # Wandb magic.
        if self.use_wandb:
            wandb.watch(self.model, log_freq=100)

        for epoch in range(total_epoch):
            with tqdm(dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{total_epoch}")

                epoch_loss = 0.
                for data, label in tepoch:
                    # Send data and label to compute device.
                    data, label = data.to(self.device), label.to(self.device)
                    
                    # Reset the gradien inside optimizer.
                    self.optim.zero_grad()

                    # Pass data to model and compute forward.
                    intermidiate, pred = self.model(to_tensor(data))
                    pred_max, pred_class_index = pred.max(dim=1)

                    self._log_progress(epoch, intermidiate, pred_class_index)

                    # Calcaulate loss between prediction and ground truth.
                    loss = self.loss_func(pred, label)

                    # Backpropagation
                    loss.backward()

                    # Gradient step
                    self.optim.step()

                    # Summarize loss and acc in every step.
                    epoch_loss += loss.item()
                    show_epoch_loss = np.round(epoch_loss, 4)
                    eval_max, eval_label = label.max(dim=1)
                    acc = (eval_label == pred_class_index).sum()/data.size(0)
                    show_acc = np.round(to_numpy(acc), 4)
                    # accsum = (eval_label == pred_class_index).sum()
                    # accnum = data.size(0)

                    # Update progress bar.
                    tepoch.set_postfix(acc=show_acc, loss=show_epoch_loss)

                if self.use_wandb: 
                    wandb.log({
                        "loss": epoch_loss,
                        "acc": acc
                    })
            
    def _predict(self, dataloader):
        # Change model to evaluation mode inplace.
        self.model.eval()
        total_pred = None

        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.to(self.device), label.to(self.device)
                intermidiate, pred = self.model(to_tensor(data))
                if torch.is_tensor(total_pred):
                    total_pred = torch.concat((total_pred, pred), 0)
                else:
                    total_pred = pred


        return total_pred
    
    def _log_progress(self, epoch, data, label):
        data_x, data_y, label = to_numpy(data[:, 0]), to_numpy(data[:, 1]), to_numpy(label)
        # self.progress_data_x.append(data_x)
        # self.progress_data_y.append(data_x)
        # self.progress_data_label.append(label)
        # self.progress_data_x = data_x
        # self.progress_data_y = data_y
        # self.progress_data_label = label
        plt.figure(figsize=(10, 10))
        plt.subplot(111)
        plt.ylabel("y")
        plt.xlabel("x")
        plt.scatter(data_x, data_y, c=label)
        if not os.path.exists("progress"):
            os.mkdir("progress")
        
        plt.savefig(os.path.join("progress", "snap_epoch_%03d.png" % epoch))

    
    def run(self, gpu_id):
        set_device(gpu_id)

        # Set seeds manually, so the result is reproducable.
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Get train and test dataset and fit into dataloader.
        self.train_dataset = CircleData(self.loader_params, self.random_state, is_train=True)
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.loader_params["batch_size"],
            shuffle=True,
        )

        self.test_dataset = CircleData(self.loader_params, self.random_state, is_train=False)
        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.loader_params["batch_size"],
            shuffle=True,
        )

        # Create model object and loss function and optimizer function.
        self.model = CircleRegressor(self.model_params).to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optim = Adam(self.model.parameters(), lr=self.model_params["lr"])

        # Train
        self._train(train_dataloader)

        # Evaluation
        train_loss = self._predict(train_dataloader)
        test_loss = self._predict(test_dataloader)

        # Calculate mse for evaluating train and test data.
        # NOTE: Not sure what's this part.
        train_score = self.loss_func(train_loss, self.train_dataset.label)
        test_score = self.loss_func(test_loss, self.test_dataset.label)

        if self.use_wandb:
            wandb.log({
                "train_score": train_score,
                "test_score": test_score
            })