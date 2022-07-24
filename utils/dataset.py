# Initialize paths
import sys
import yaml
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from torch.utils.data import Dataset

PATHS = yaml.safe_load(open("paths.yaml"))
for k in PATHS: sys.path.append(PATHS[k])

class CircleData(Dataset):
    DEFAULTS = {}
    def __init__(self, config):
        self.__dict__.update(self.DEFAULTS, **config)

        self.data, self.label = make_circles(
            n_samples=self.N_data,
            shuffle=True,
            noise=0.075,
            factor=0.5
        )

        self.data = self.data.astype("float32")
        self.label = self.label.astype("float32")

    
    def plot(self):
        plt.title("Circle Data")
        x = self.data[:, 0]
        y = self.data[:, 1]
        plt.scatter(x, y, c=self.label)
        plt.show()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]