import numpy as np
import torch

def mse(pred, label):
    assert pred.shape == label.shape
    if torch.is_tensor(pred):
        return torch.mean((pred-label)**2)
    return np.mean((pred-label)**2)