import torch.nn as nn
import torch.nn.functional as F

class MnistMlpModel(nn.Module):
    def __init__(self, config):
        super(MnistMlpModel, self).__init__()
        self.__dict__.update({}, **config)

        layers = []

        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(self.dropout_rate))

        for n_layer in range(self.n_layers):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout_rate))
        
        layers.append(nn.Linear(self.hidden_size, self.output_size))

        self.seq = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.seq(x)
        return F.log_softmax(x, dim=1)