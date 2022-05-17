#!/usr/bin/python3
import torch
import torch.nn as nn

x = torch.tensor([]) 
y = torch.tensor([]) 

for i in range(9):
    x = torch.cat((x, torch.tensor([[i]])), 0)
    y = torch.cat((y, torch.tensor([[i * 2]])), 0)


print("X: ", x.numpy().tolist())
print("Y: ", y.numpy().tolist())

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        x = self.linear(x)
        return x

network = Network()

epochs = 1000
mseloss = nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr = 0.03)
all_losses = []
current_loss = 0
plot_every = 50

for epoch in range(epochs):

  # input training example and return the prediction
  yhat = network.forward(x)

  # calculate MSE loss
  loss = mseloss(yhat, y)
  
  # backpropogate through the loss gradiants
  loss.backward()

  # update model weights
  optimizer.step()

  # remove current gradients for next iteration
  optimizer.zero_grad()

  # append to loss
  current_loss += loss
  if epoch % plot_every == 0:
      all_losses.append(current_loss / plot_every)
      current_loss = 0
  
  # print progress
  if epoch % 500 == 0:
    print(f'Epoch: {epoch} completed')


inp = torch.tensor([[[100]]]).type(torch.float);
out = network(inp)
print(out)

def save_weights(model, filename='weights.json'):
    import json
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

save_weights(network)
