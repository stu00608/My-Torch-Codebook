import torch
import torch.nn as nn
import yaml
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.circle import CircleRegressor
from utils.helper import set_device
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
    
    def _train(self, dataloader):
        """Training the model"""
        # Set the model to training mode inplace.
        self.model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_epoch = self.model_params["N_epoch"]
        training_loss = []

        for epoch in range(total_epoch):
            with tqdm(dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{total_epoch}")

                epoch_loss = 0.
                for data, label in tepoch:
                    # Send data and label to compute device.
                    data, label = data.to(device), label.to(device)
                    
                    # Reset the gradien inside optimizer.
                    self.optim.zero_grad()

                    # Pass data to model and compute forward.
                    pred = self.model(data)

                    # Calcaulate loss between prediction and ground truth.
                    loss = self.loss_func(pred, label)

                    # Backpropagation
                    loss.backward()

                    # Gradient step
                    self.optim.step()

                    # Summarize loss in every step.
                    epoch_loss += loss.item()

                    # Update progress bar.
                    tepoch.set_postfix(loss=epoch_loss)
                
                training_loss.append(epoch_loss)
        
        return training_loss
    
    def _predict(self, dataloader):
        # Change model to evaluation mode inplace.
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_pred = []

        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.to(device), label.to(device)
                pred = self.model(data)
                total_pred.append(pred.numpy())
        
        total_pred = np.concatenate(total_pred, axis=0)

        return total_pred
    
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
        self.model = CircleRegressor(self.model_params)
        self.loss_func = mse
        self.optim = Adam(self.model.parameters(), lr=self.model_params["lr"])

        # Train
        training_loss = self._train(train_dataloader)

        # Evaluation
        train_pred = self._predict(train_dataloader)
        test_pred = self._predict(test_dataloader)

        # Calculate mse for evaluating train and test data.
        train_score = mse(train_pred, self.train_dataset.label)
        test_score = mse(test_pred, self.test_dataset.label)

        # Visualize the result.
        plt.title(f"Circle loss, train score : {train_score}, test_score : {test_score}")
        plt.plot(list(range(len(training_loss))), training_loss)
        plt.show()