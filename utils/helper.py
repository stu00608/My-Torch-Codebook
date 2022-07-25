import os
import torch
import numpy as np

def set_device(gpu_id):
    # Manage GPU availability
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    if gpu_id != "": 
        torch.cuda.set_device(0)
        
    else:
        n_threads = torch.get_num_threads()
        n_threads = min(n_threads, 8)
        torch.set_num_threads(n_threads)
        print("Using {} CPU Core".format(n_threads))

def one_hot(inputs, num_classes):
  return np.squeeze(np.eye(num_classes)[inputs.reshape(-1)])

def to_numpy(x):
    # x is already a numpy array
    if type(x) != type(torch.tensor(0)): 
        return x
    
    if x.is_cuda:
        return x.data.cpu().numpy()
    return x.data.numpy()

def to_tensor(x):
    if type(x) != type(torch.tensor(0)): 
        x = torch.tensor(x)
    if torch.cuda.is_available():
        return x.cuda()
    return x