import os
import sys
import yaml
import torch
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from datasets.mnist_dataset import mnist_dataset
from models.mnist_model import MnistAutoencoderModel, MnistAutoencoderCnnModel
from utils.files import file_choices
from utils.helper import set_device, to_numpy, to_tensor

PATHS = yaml.safe_load(open("paths.yaml"))
for k in PATHS:
    sys.path.append(PATHS[k])


class MnistAutoencoderCnnSolver:
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
        # Create a folder to save model weight.
        if not os.path.exists("weights"):
            os.makedirs("weights")

        if self.use_wandb:
            wandb.watch(self.model, log_freq=100)

        total_epoch = self.model_params["N_epoch"]
        for epoch in range(total_epoch):
            self.model.train()
            with tqdm(dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{total_epoch}")

                epoch_loss = []
                for index, (data, label) in enumerate(tepoch):

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

                    epoch_loss.append(to_numpy(loss))

                    # Calculate average loss in this epoch
                    current_avg_loss = np.mean(epoch_loss)
                    # Update progress bar.
                    tepoch.set_postfix(loss=current_avg_loss)

                if self.use_wandb:
                    wandb.log({
                        "train_loss": epoch_loss,
                    })
                state_dict_path = os.path.join("weights", self.run_name + f"_Epoch{(epoch+1):03}.pth")

                torch.save(self.model.state_dict(), state_dict_path)

    def _test(self, dataloader):
        self.model.eval()

        # If there is no weights folder then throw an exception.
        if not os.path.exists("weights"):
            raise RuntimeError(
                "weights folder not exist, please run training first.")

        test_loss = 0.
        with torch.no_grad():

            for i, (data, label) in enumerate(dataloader):

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
    
    def _visualize(self, dataloader: DataLoader):
        plt.figure(figsize=(16, 5))
        labels = dataloader.dataset.targets.numpy()
        n = 10
        t_idx = {i:np.where(labels==i)[0][0] for i in range(n)}
        for i in range(n):
            ax = plt.subplot(2, n, i+1)
            img = dataloader.dataset[t_idx[i]][0].unsqueeze(0).to(self.device)

            self.model.eval()
            with torch.no_grad():
                rec_img_vector  = self.model.decoder(self.model.encoder(img))

            rec_img = to_numpy(rec_img_vector).reshape(28, 28, 1)
            plt.imshow(img.cpu().squeeze().numpy(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title('Original images')

            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(rec_img, cmap='gray')  
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title('Reconstructed images')
        plt.show()   

    def run(self, is_inference=False, visualize=False):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        train_dataloader, test_dataloader = mnist_dataset(self.loader_params)

        self.model = MnistAutoencoderCnnModel(self.model_params).to(self.device)
        self.loss = nn.MSELoss().to(self.device)
        self.optimizer = Adam(self.model.parameters(),
                              lr=self.model_params["lr"])

        if not is_inference:
            self._train(train_dataloader)
        
        if visualize:
            self._visualize(train_dataloader)
        # self._test(test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=lambda s: file_choices(("yaml"), s), required=True)
    parser.add_argument('--inference', action="store_true")
    parser.add_argument('--gpu_id', type=str, default="")
    parser.add_argument('--visualize', action="store_true")
    args = parser.parse_args()

    solver = MnistAutoencoderCnnSolver(args.config, args.gpu_id)
    solver.run(is_inference=args.inference, visualize=args.visualize)
