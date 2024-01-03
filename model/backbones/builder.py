import torch
import torch.nn as nn



'''build activation functions'''
def BuildActivation(activation_type, **kwargs):
    supported_activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'relu6': nn.ReLU6,
        'prelu': nn.PReLU,
        'sigmoid': nn.Sigmoid,
    }
    assert activation_type in supported_activations, 'unsupport activation type %s...' % activation_type
    return supported_activations[activation_type](**kwargs)