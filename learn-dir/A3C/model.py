import torch
import torch.nn as nn
import numpy as np

'''
Implementation of A3C-LSTM 
- Check paper pg 12: 
    - convolutional layer 1 = 16 filters, 8x8 kernel, 4 stride
    - convolutional layer 2 = 32 filters, 4x4 kernel, 2 stride
    - linear layer = 256 hidden units
    - All hidden layers have ReLU
    - all hiddden layers were followed by recitifier non-linearity
    - additional 256 cell LSTM layer
'''
class A3C(nn.Module):
    def __init__(self,input_shape, layer1, kernel_size1, stride1, layer2, kernel_size2, stride2, fc1_dim, lstm_dim, out_actor_dim, out_critic_dim):
        super(A3C, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=input_shape, out_channels=layer1, kernel_size=kernel_size1, stride=stride1)
        self.conv2 = torch.nn.Conv2d(in_channels=layer1, out_channels=layer2, kernel_size=kernel_size2, stride=stride2)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_features=32*9*9, out_features=fc1_dim)
        self.out_actor = torch.nn.Linear(in_features=lstm_dim, out_features=out_actor_dim)
        self.out_critic = torch.nn.Linear(in_features=lstm_dim, out_features=out_critic_dim)
        #lstm cell
        self.lstm_cell = torch.nn.LSTMCell(fc1_dim, lstm_dim)
        
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        for name, param in self.lstm_cell.named_parameters():
            if 'bias' in name:
                param.data.zero_()
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.out_critic.weight)
        self.out_critic.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.out_actor.weight)
        self.out_actor.bias.data.zero_()

    def forward(self, x):
        x, (hx, cx) = x
        out_backbone = self.conv1(x)
        out_backbone = nn.ReLU(out_backbone)
        out_backbone = self.conv2(out_backbone)
        out_backbone = nn.ReLU(out_backbone)
        out = out_backbone.reshape(-1,32*9*9)
        out = self.fc1(out)
        out = self.relu(out)
        #lstm cell
        hx, cx = self.lstm_cell(out, (hx, cx))
        out = hx
        #actor
        actor = self.out_actor(out)
        #critic
        critic = self.out_critic(out)
        
        return actor,critic,(hx, cx)

