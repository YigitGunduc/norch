import json
import torch

def save_weights(model, filename='weights.json'):

    weights = []
    biases = []

    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            weights.append(layer.state_dict()['weight'].detach().numpy().tolist())
            biases.append(layer.state_dict()['bias'].detach().numpy().tolist())

            # TODO: implement a check to verify bias is used 
            # if (layer.state_dict()['bias']):


    model_params = {'weights': weights, 
                    'biases': biases}

    with open(filename, "w") as outfile:
        json.dump(model_params, outfile)
    
    return weights, biases

def get_weights(model):

    weights = []
    biases = []

    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            weights.append(layer.state_dict()['weight'].detach().numpy().tolist())
            biases.append(layer.state_dict()['bias'].detach().numpy().tolist())

    return weights, biases
