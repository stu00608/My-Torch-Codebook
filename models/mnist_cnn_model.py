import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistCnnModel(nn.Module):
    def __init__(self, config):
        super(MnistCnnModel, self).__init__()
        self.__dict__.update({}, **config)

        self.conv1 = nn.Conv2d(1, 32, config["kernel_size"], config["stride_size"])
        self.conv2 = nn.Conv2d(32, 64, config["kernel_size"], config["stride_size"])
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.fc1 = nn.Linear(9216, config["hidden_size"])
        self.fc2 = nn.Linear(config["hidden_size"], config["output_size"])
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
        

