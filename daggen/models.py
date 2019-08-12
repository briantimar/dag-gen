""" Models for parameterizing distributions over graphs. """

import torch
from torch.distributions.categorical import Categorical

class TorchDAG:
    """ Define and apply batched computational graphs based on torch mask tensors and a list of activation functions."""

    def __init__(self, activation_choices, connections, activations, active_vertices):
        """ activation_choices: a list of torch activation functions
            connections: (N, max_vertices, max_vertices) tensor defining connections between neurons
            activations: (N, max_vertices) tensor specifying which activations are to be applied at the output of each vertex.
            active_vertices: (N, max_vertices) unit8 tensor specifying how many vertices of each graph are used.
        """
        self.activation_choices = activation_choices
        self.num_activations = len(activation_choices)
        self.connections = connections
        self.activations = activations
        self.active_vertices = active_vertices

        self.max_vertices = self.activations.size(1)
        self.batch_size = self.activations.size(0)

        if tuple(connections.shape) != (self.batch_size, self.max_vertices, self.max_vertices):
            raise ValueError("Invalid connections shape {0}".format(connections.shape))
        if tuple(activations.shape) != (self.batch_size, self.max_vertices):
            raise ValueError("Invalid activations shape {0}".format(activations.shape))
        if tuple(active_vertices.shape) != (self.batch_size, self.max_vertices):
            raise ValueError("Invalid active_vertices shape {0}".format(active_vertices.shape))

        nact = self.active_vertices[:, 0].sum().item()
        if nact != self.batch_size:
            raise ValueError("All initial vertices should be active")

    def forward(self, x):
        """ Compute forward passes for each of the networks.
            `x`: (N,) tensor 
            Returns: (N,), obtained by applying the ith network to the ith element of x.
            """
        if tuple(x.shape) != (self.batch_size,):
            raise ValueError("Invalid input shape {0} for DAG batch size {1}".format(x.shape, self.batch_size))

        # tensor to hold intermediate computation results
        # y[:, i] is the graph value at layer i of the topological sort, or zero where the graph computation has already finished
        y = torch.zeros(self.batch_size, self.max_vertices, dtype=torch.float)
        inputs = torch.zeros_like(y)
        
        for i in range(self.max_vertices):
            if i == 0:
                summed_input = x
            else:
                input_vertices = self.connections[:, i, :]
                inputs.copy_(y)
                inputs[~input_vertices] = 0
                summed_input = inputs.sum(dim=1)
            
            all_act = torch.stack([f(summed_input) for f in self.activation_choices], dim=1)
            output = all_act[range(self.batch_size), self.activations[:, i]]
            y[self.active_vertices[:, i], i] = output[self.active_vertices[:, i]]
            if i > 0:
                y[~self.active_vertices[:, i], i] = y[~self.active_vertices[:, i], i-1]
        return y[:, -1]
            


        

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

class TwoLayerMLP(MLP):
    """ Two linear applications, ie a single hidden layer. """

    def __init__(self, input_size, hidden_size, output_size, activation=torch.nn.ReLU):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        super().__init__([input_size, hidden_size, output_size], activation=activation)

class GraphRNN(torch.nn.Module):
    """An autoregressive graph model a la https://arxiv.org/abs/1802.08773."""

    def __init__(self, vertex_cell, edge_cell, activation_cell, 
                    vertex_logits, edge_logits, activation_logits):
        """ `vertex_cell`: the `backbone` of the graph generator. Applied once for each vertex in the graph. Takes as input vertex hidden state and edge hidden state
        from the previous vertex, outputs new hidden state.
            `edge_cell`: computes probabilities for a particular vertex to be connected to its predecessors. Takes as input edge hidden state, and binary adjacency value from a 
            previous vertex; computes new edge hidden state. 
            `activation_cell`: takes as input edge hidden state, and a binary adjacency value; computes new edge hidden state.
            `vertex_logits`: takes as input vertex hidden state, returns two logits indicating whether a new vertex should be added (not adding a vertex corresponds to 'end of sequence')
            `edge_logits`: takes as input edge hidden state, returns logits for determining connectivity (two classes, connected or unconnected).
            `activation_logits`: takes as input edge hidden state, returns logits for determining activation function to be applied"""
        
        super().__init__()

        self.add_module('vertex_cell', vertex_cell)
        self.add_module('edge_cell', edge_cell)
        self.add_module('activation_cell', activation_cell)
        self.add_module('vertex_logits', vertex_logits)
        self.add_module('edge_logits', edge_logits)
        self.add_module('activation_logits', activation_logits)



class ScalarGraphGRU(GraphRNN):
    """ Graph generator which uses GRU cells."""

    def __init__(self, hidden_size, logits_hidden_size, num_activations):
        """ hidden_size: dimensionality of the hidden state for vertex and edge cells.
            logits_hidden_size: size of the hidden layer used in computing logits from hidden state
            num_activations: the number of possible activation functions. """

        self.hidden_size = hidden_size
        self.logits_hidden_size = logits_hidden_size
        self.num_activations = num_activations
        self.vertex_input_size = hidden_size + 1

        vertex_cell = torch.nn.GRUCell(self.vertex_input_size, hidden_size)
        edge_cell = torch.nn.GRUCell(1, hidden_size)
        activation_cell = torch.nn.GRUCell(1, hidden_size)
        vertex_logits = TwoLayerMLP(hidden_size, logits_hidden_size, 2)
        edge_logits = TwoLayerMLP(hidden_size, logits_hidden_size, 2)
        activation_logits = TwoLayerMLP(hidden_size, logits_hidden_size, num_activations)

        super().__init__(vertex_cell, edge_cell, activation_cell, 
                        vertex_logits, edge_logits, activation_logits)

        ## initialization of the hidden state
        self.hidden_init = torch.nn.Parameter(torch.randn(hidden_size))
        self.vertex_input_init = torch.nn.Parameter(torch.randn(self.vertex_input_size))

    def _get_samples_and_log_probs(self, logits):
        """ Obtain samples and log-probabilities, using a Categorical distribution, for the given logits."""
        dist = Categorical(logits=logits)
        samples = dist.sample()
        lps = dist.log_prob(samples)
        return samples, lps

    def sample_graph_tensors(self, N, max_vertices=None, min_vertices=None ):
        """ Sample N graph encodings from the ScalarGraphGRU.
        
        `N`: how many graphs to sample
        `max_vertices`: if not None, the max number of vertices to permit in each graph.
        `min_vertices`: if not None, the sampled graphs will contain at least this many vertices.
        
        Returns: next_active_vertices (N, maxnum) bool tensor specifying which vertices exist in each graph.
                connections_all (N, maxnum, maxnum) bool tensor. The i, j, k element is nonzero iff a connection k -> j exists in the ith graph.
                activations (N, maxnum) int tensor specifying an activation function to be applied at each vertex."""

        if max_vertices is not None and min_vertices is not None and (max_vertices < min_vertices):
            raise ValueError("Invalid vertex number specs: min, max = {0}, {1}".format(min_vertices, max_vertices))
        if max_vertices is not None and max_vertices < 1:
            raise ValueError("Invalid max vertex number {0}".format(max_vertices))

        vertex_index = 0
        graph_complete = False
        vertex_hidden_state = self.hidden_init.view(1, -1).expand(N, self.hidden_size)
        vertex_input = self.vertex_input_init.view(1, -1).expand(N, self.vertex_input_size)

        log_probs = []

        next_active_vertices = []
        all_connections = []
        activations = []

        #keep track of which graphs are still being sampled
        next_active_graphs = torch.ones(N,dtype=torch.uint8)
        
        while not graph_complete:
            #compute new hidden state
            vertex_hidden_state = self.vertex_cell(vertex_input, vertex_hidden_state)
            
            #check whether another vertex should be added
            vertex_logits = self.vertex_logits(vertex_hidden_state)
            add_vertex, lp_vertex = self._get_samples_and_log_probs(vertex_logits)
            #a graph is active only if it has added a vertex at each sampling step.
            # these are the graphs for which another vertex will be added after connections have been determined for the current vertex
            next_active_graphs = next_active_graphs & (add_vertex > 0)

            next_active_vertices.append(next_active_graphs.data)

            #now pass hidden state down to edge cells
            edge_hidden_state = vertex_hidden_state
            edge_input_init = torch.zeros(N,1)
            edge_input = edge_input_init

            #determine whether to connect to each previous vertex
            connections = []
            for prev_index in range(vertex_index):
                #compute edge hidden state
                edge_hidden_state = self.edge_cell(edge_input, edge_hidden_state)            
                #sample connectivity to prev_index
                connection_logits = self.edge_logits(edge_hidden_state)
                add_connection, lp_connection = self._get_samples_and_log_probs(connection_logits)
                lp_vertex += lp_connection
                connections.append(add_connection)

                edge_input = add_connection.view(N, 1).to(dtype=torch.float)
            
            all_connections.append(connections)
            # finally, determine which activation function to apply to this vertex
            act_state = self.activation_cell(edge_input, edge_hidden_state)
            act_logits = self.activation_logits(act_state)
            activation, lp_act = self._get_samples_and_log_probs(act_logits)
            lp_vertex += lp_act

            activations.append(activation)
            log_probs.append(lp_vertex)

            #compute input to the next vertex cell
            vertex_input = torch.cat((edge_hidden_state, activation.view(-1, 1).to(dtype=torch.float)), dim=1)
            
            vertex_index += 1
            ##sampling halts when all graph sequences have terminated, or max number of vertices is reached
            min_vertices_satisfied = (min_vertices is None or vertex_index >= min_vertices)
            max_verticies_satisfied = (max_vertices is not None and vertex_index == max_vertices)
            graph_complete = min_vertices_satisfied and ( next_active_graphs.sum().item()==0 or max_verticies_satisfied)

        #when all graphs have finished sampling, stack together the sequences that define the graphs, and the corresponding log-probs

        #(N, max_num_vertices) tensor specifying the selected activations
        #NOTE this gets stacked along dim 1 because the each entry in the activations list corresponds to a single vertex (and has length N)
        activations = torch.stack(activations,dim=1)

        # (N, max_num_vertices) tensor specifying whether a vertex will be added at each step
        # once a vertex is not added, the graph is dead
        next_active_vertices = torch.stack(next_active_vertices, dim=1)      
        #same thing, but now indicating whether a graph is active at step i
        active_vertices = torch.ones_like(next_active_vertices)
        active_vertices[:, 1:] = next_active_vertices[:, :-1]
        
        #number of vertices in the largest graph of the batch
        max_num_vertices = activations.size(1)

        #bool tensor which defines the connectivity of each graph
        #the i, j, k entry indicates whether a connection k ->j exists in the ith graph
        connections_all = torch.zeros(N, max_num_vertices, max_num_vertices, dtype=torch.uint8)
        
        #log-probabilities for each graph
        log_probs = torch.stack(log_probs, dim=1)
        # graph sampling stops when first "end of sequence" is reached.
        log_probs[~active_vertices] = 0

        for i in range(1,max_num_vertices):
            # for each graph, lookup which connections flow into the ith neuron
            #(N, i) tensor indicating connections to previous tensors
            conns = torch.stack(all_connections[i], dim=1)
            connections_all[:, i, :i] = conns
        
        return next_active_vertices, connections_all, activations, log_probs.sum(dim=1)