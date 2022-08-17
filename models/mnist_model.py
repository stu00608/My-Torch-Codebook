import torch
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


class MnistCnnModel(nn.Module):
    def __init__(self, config):
        super(MnistCnnModel, self).__init__()
        self.__dict__.update({}, **config)

        self.conv1 = nn.Conv2d(
            1, 32, config["kernel_size"], config["stride_size"])
        self.conv2 = nn.Conv2d(
            32, 64, config["kernel_size"], config["stride_size"])
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


class MnistAutoencoderModel(nn.Module):

    def __init__(self, config):
        super(MnistAutoencoderModel, self).__init__()
        self.__dict__.update({}, **config)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.z_dim),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.input_size),
            nn.ReLU(True)
        )

    def forward(self, x):
        h = self.encoder(x)
        x = self.decoder(h)

        return x, h