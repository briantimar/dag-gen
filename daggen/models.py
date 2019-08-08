""" Models for parameterizing distributions over graphs. """

import torch

class MLP(torch.nn.Module):
    """Dense network."""

    def __init__(self, layer_sizes, activation=torch.nn.ReLU):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation()
        self.layers = []
        for i in range(len(layer_sizes)-1):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.add_module('layer_%d'%i, layer)
            self.layers.append(layer)
        self.num_layers = len(self.layers)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers -1:
                x = self.activation(x)
        return x



class GraphRNN(torch.nn.Module):
    """An autoregressive graph model a la https://arxiv.org/abs/1802.08773."""

    def __init__(self, vertex_cell, edge_cell, activation_cell, 
                    edge_logits, activation_logits):
        """ `vertex_cell`: the `backbone` of the graph generator. Applied once for each vertex in the graph. Takes as input vertex hidden state and edge hidden state
        from the previous vertex, outputs new hidden state.
            `edge_cell`: computes probabilities for a particular vertex to be connected to its predecessors. Takes as input edge hidden state, and binary adjacency value from a 
            previous vertex; computes new edge hidden state. 
            `activation_cell`: takes as input edge hidden state, and a binary adjacency value; computes new edge hidden state.
            `edge_logits`: takes as input edge hidden state, returns logits for determining connectivity (two classes, connected or unconnected).
            `activation_logits`: takes as input edge hidden state, returns logits for determining activation function to be applied"""
        
        super().__init__()
        # self.vertex_cell = vertex_cell
        # self.edge_cell = edge_cell
        # self.activation_cell = activation_cell
        # self.edge_logits = edge_logits
        # self.activation_logits = activation_logits
        self.add_module('vertex_cell', vertex_cell)
        self.add_module('edge_cell', edge_cell)
        self.add_module('activation_cell', activation_cell)
        self.add_module('edge_logits', edge_logits)
        self.add_module('activation_logits', activation_logits)


class GraphGRU(GraphRNN):
    """ Graph generator which uses GRU cells."""

    def __init__(self, hidden_size):
        vertex_cell = torch.nn.GRUCell(hidden_size, hidden_size)
        edge_cell = torch.nn.GRUCell(1, hidden_size)
        activation_cell = torch.nn.GRUCell(1, hidden_size)
        edge_logits = torch.nn.Lin

    

