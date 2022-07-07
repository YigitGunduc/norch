import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


input_size = 784 # 28x28
hidden_size = 32 
num_classes = 10
num_epochs = 10
batch_size = 128
learning_rate = 0.001

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


model = torch.load('model.pt')
model = model.cpu()

import numpy as np

inp = np.zeros((1, 28, 28))
inp = inp.reshape((1, 28 * 28))
out = model.forward(torch.tensor(inp).type(torch.float))
print(out[0])
