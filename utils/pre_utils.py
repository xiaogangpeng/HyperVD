from collections import defaultdict
import os
import pickle
import json
import torch.nn as nn
import torch as th
import torch.optim as optim
import numpy as np
import random
import math
import subprocess
import random

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

def th_dot(x, y, keepdim=True):
    return th.sum(x * y, dim=1, keepdim=keepdim)

def pad_sequence(data_list, maxlen, value=0):
    return [row + [value] * (maxlen - len(row)) for row in data_list]

def normalize_weight(adj_mat, weight):
    degree = [1 / math.sqrt(sum(np.abs(w))) for w in weight]
    for dst in range(len(adj_mat)):
        for src_idx in range(len(adj_mat[dst])):
            src = adj_mat[dst][src_idx]
            weight[dst][src_idx] = degree[dst] * weight[dst][src_idx] * degree[src]

def nn_init(nn_module, method='orthogonal'):
    """
    Initialize a Sequential or Module object
    Args:
        nn_module: Sequential or Module
        method: initialization method
    """
    if method == 'none':
        return
    for param_name, _ in nn_module.named_parameters():
        if isinstance(nn_module, nn.Sequential):
            # for a Sequential object, the param_name contains both id and param name
            i, name = param_name.split('.', 1)
            param = getattr(nn_module[int(i)], name)
        else:
            param = getattr(nn_module, param_name)
        if param_name.find('weight') > -1:
            init_weight(param, method)
        elif param_name.find('bias') > -1:
            nn.init.uniform_(param, -1e-4, 1e-4)

def get_params(params_list, vars_list):
    """
    Add parameters in vars_list to param_list
    """
    for i in vars_list:
        if issubclass(i.__class__, nn.Module):
            params_list.extend(list(i.parameters()))
        elif issubclass(i.__class__, nn.Parameter):
            params_list.append(i)
        else:
            print("Encounter unknown objects")
            exit(1)

def categorize_params(args):
    """
    Categorize parameters into hyperbolic ones and euclidean ones
    """
    stiefel_params, euclidean_params = [], []
    get_params(euclidean_params, args.eucl_vars)
    get_params(stiefel_params, args.stie_vars)
    return stiefel_params, euclidean_params

def get_activation(args):
    if args.activation == 'leaky_relu':
        return nn.LeakyReLU(args.leaky_relu)
    elif args.activation == 'rrelu':
        return nn.RReLU()
    elif args.activation == 'relu':
        return nn.ReLU()
    elif args.activation == 'elu':
        return nn.ELU()
    elif args.activation == 'prelu':
        return nn.PReLU()
    elif args.activation == 'selu':
        return nn.SELU()

def init_weight(weight, method):
    """
    Initialize parameters
    Args:
        weight: a Parameter object
        method: initialization method 
    """
    if method == 'orthogonal':
        nn.init.orthogonal_(weight)
    elif method == 'xavier':
        nn.init.xavier_uniform_(weight)
    elif method == 'kaiming':
        nn.init.kaiming_uniform_(weight)
    elif method == 'none':
        pass
    else:
        raise Exception('Unknown init method')

