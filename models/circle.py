import torch.nn as nn

class CircleRegressor(nn.Module):
    """This is a regression model for predicting inner or outer circle. 
    Inherit from `torch.nn.Module`."""

    def __init__(self, config):
        super(CircleRegressor, self).__init__()
        self.__dict__.update({}, **config)
        layers = []
        
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=self.dropout_rate))

        for _ in range(self.n_layers):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=self.dropout_rate))

        self.seq = nn.Sequential(*layers)

        self.out = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        h = self.seq(x)
        out = self.out(h)
        return out.squeeze(-1)