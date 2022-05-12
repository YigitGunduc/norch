import torch
import json
import random
import numpy as np
import torch.nn.functional as F
from torch import nn

class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.l = nn.Linear(3, 5)
        self.l2 = nn.Linear(5, 3)
        self.l3 = nn.Linear(3, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # forward propagate input
    def forward(self, x):
        x = self.l(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.sigmoid(x)
        return x

mlp = MLP(5)

def save_weights(model, filename='weights.json'):
    weights = []
    biases = []
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            weights.append(layer.state_dict()['weight'].detach().numpy().tolist())
            biases.append(layer.state_dict()['bias'].detach().numpy().tolist())


    model_params = {'weights': weights, 
                    'biases': biases}

    with open(filename, "w") as outfile:
        json.dump(model_params, outfile)
    
    return weights, biases

  
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

inp = [1, 2, 3]
x = torch.ones(3)
print(mlp.forward(x))

weights, biases = save_weights(mlp)

x = x.numpy()
for weight, bias in zip(weights, biases):
    x = np.matmul(x, np.array(weight).T) + bias
    # x = relu(x)

# x = sigmoid(x)
print(x)
print('done')
