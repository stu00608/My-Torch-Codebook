# Initialize paths
import sys
import yaml
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from utils.helper import one_hot


PATHS = yaml.safe_load(open("paths.yaml"))
for k in PATHS: sys.path.append(PATHS[k])

class CircleData(Dataset):
    """This dataset provides inner and outer 2-D point for classification. 
    Inherit from `torch.utils.data.Dataset`
    
    Methods
    -------
    plot()
        Plot the circle .
    """


    def __init__(self, config, random_state, is_train: bool):
        self.__dict__.update({}, **config)

        raw_data, raw_label = make_circles(
            n_samples=self.N_data,
            shuffle=True,
            noise=self.circle_noise,
            factor=self.circle_factor,
            random_state=random_state
        )

        raw_label = one_hot(raw_label, 2)

        x_train, x_test, y_train, y_test = train_test_split(
            raw_data, 
            raw_label,
            train_size=1-self.test_split,
            test_size=self.test_split,
            random_state=random_state,)

        if is_train:
            self.data = x_train.astype("float32")
            self.label = y_train.astype("float32")
        else:
            self.data = x_test.astype("float32")
            self.label = y_test.astype("float32")
    
    def plot(self):
        """Plot the circle."""
        plt.title("Circle Data")
        x = self.data[:, 0]
        y = self.data[:, 1]
        plt.scatter(x, y, c=self.label)
        plt.show()
    
    def __len__(self):
        """Returns the length of this dataset."""
        return len(self.data)
    
    def __getitem__(self, index):
        """Get items from this dataset. Used in training or inferencing process."""
        return self.data[index], self.label[index]