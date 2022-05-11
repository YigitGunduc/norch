import torch
from torch import nn

class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer = nn.Linear(n_inputs, 1)
        self.activation = nn.Sigmoid()

    # forward propagate input
    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x

mlp = MLP(5)


x = (torch.nn.ParameterList(mlp.parameters()))
def get_params(params):
  out = []
  
  params = (torch.nn.ParameterList(mlp.parameters()))
  for param in params:
    out.append(param.detach().numpy())

  return out[0]
    
  
print(get_params(mlp.parameters()))

print('done')