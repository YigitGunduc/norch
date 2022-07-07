def save_weights(model, device='cpu', filename='weights.json'):

    # author: @yigitgunduc
    # github: https://github.com/YigitGunduc/norch
    # functin to save torch weights as json
    # requires:
    #   - model: torch model to save - must consist of linear layers only
    #   - device: device where it model lives(cpu/cuda) - defaults is cpu - optional 
    #   - filename: name of the weight file - default weights.json - optional 

    import json
    import torch

    weights = []
    biases = []

    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            if (device == 'cpu'):
                weights.append(layer.state_dict()['weight'].detach().numpy().tolist())
                biases.append(layer.state_dict()['bias'].detach().numpy().tolist())
            else:
                weights.append(layer.state_dict()['weight'].detach().cpu().numpy().tolist())
                biases.append(layer.state_dict()['bias'].detach().cpu().numpy().tolist())


    model_params = {'weights': weights, 
                    'biases': biases}

    with open(filename, "w") as outfile:
        json.dump(model_params, outfile)


import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO]:found device:{device}")

input_size = 784 # 28x28
hidden_size = 32 
num_classes = 10
num_epochs = 10
batch_size = 128
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor()
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
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

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


save_weights(model, device)
torch.save(model, 'model.pt')
