import os
import torch

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