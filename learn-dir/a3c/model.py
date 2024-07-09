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
    def __init__(self, input_shape, layer1, kernel1, stride1, layer2, kernel2, stride2, fc1_dim, lstm_dim, out_actor_dim, out_critic_dim):
        super(A3C, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, layer1, kernel1, stride1)
        self.conv2 = nn.Conv2d(layer1, layer2, kernel2, stride2)
        self.fc1 = nn.Linear(32*9*9, fc1_dim)
        self.lstm_cell = nn.LSTMCell(fc1_dim, lstm_dim)
        self.out_actor = nn.Linear(lstm_dim, out_actor_dim)
        self.out_critic = nn.Linear(lstm_dim, out_critic_dim)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()
        
        for name, param in self.lstm_cell.named_parameters():
            if "bias" in name:
                param.data.zero_()
            elif "weight" in name:
                nn.init.xavier_uniform_(param)
        
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.out_critic.weight)
        self.out_critic.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.out_actor.weight)
        self.out_actor.bias.data.zero_()

    def forward(self, x):
        x, (hx, cx) = x
        out1 = nn.ReLU(self.conv1(x))
        out1 = nn.ReLU(out1)
        out2 = out1.reshape(-1, 32*9*9)
        out2 = nn.ReLU(self.fc1(out2))
        #lstm
        hx, cx = self.lstm_cell(out2, (hx, cx))
        out2 = hx
        #actor
        actor = self.out_actor(out2)
        #critic
        critic = self.out_critic(out2)

        return actor, critic, (hx, cx)


