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
    

def get_weights(model):
    import json
    import torch

    weights = []
    biases = []

    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            weights.append(layer.state_dict()['weight'].detach().numpy().tolist())
            biases.append(layer.state_dict()['bias'].detach().numpy().tolist())

    return weights, biases
