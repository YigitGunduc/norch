import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

device = 'cpu'

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)



def save_weights(model, filename='weights.json'):
    import json
    import torch

    weights = []
    biases = []

    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            weights.append(layer.state_dict()['weight'].detach().numpy().tolist())
            biases.append(layer.state_dict()['bias'].detach().numpy().tolist())


    model_params = {'weights': weights, 
                    'biases': biases}

    with open(filename, "w") as outfile:
        json.dump(model_params, outfile)

save_weights(model)

with torch.no_grad():
    inp = np.zeros((1, 784))
    inp = torch.tensor(inp).type(torch.float)
    print(inp.shape)
    out = model(inp)
    print(out.data)
