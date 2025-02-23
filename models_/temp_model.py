import torch
import torch.nn as nn
import torch.nn.functional as f
class CNNModel(nn.Module):

    def __init__(self):
        """
        The CNN consists of three convolutional layers, one fully connected layer and one output layer with 4 nodes.
        A kernel of size 5 with stride 1 is applied in each convolution layers. The first two convolutional layer are
        followed by a max pooling layer with kernel size 2 and stride 2.
        """
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=0
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=0
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            stride=1,
            padding=0
        )


        self.fc1 = nn.Linear(
            in_features=32768,
            out_features=2046
        )

        self.fc2 = nn.Linear(
            in_features=2046,
            out_features=4
        )

    def forward(self, x):
        if torch.any(torch.isnan(x)):
            print("NaN detected in input")
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, kernel_size=2, stride=2)
        if torch.any(torch.isnan(x)):
            print("NaN detected in layer 1")
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, kernel_size=2, stride=2)
        if torch.any(torch.isnan(x)):
            print("NaN detected in layer 2")
        x = f.relu(self.conv3(x))
        x = f.max_pool2d(x, kernel_size=2, stride=2)
        if torch.any(torch.isnan(x)):
            print("NaN detected in layer 3")
        x = x.view(x.shape[0], -1)
        x = f.dropout(f.relu(self.fc1(x)), p=0.5, training=self.training)
        x = self.fc2(x)

        return x