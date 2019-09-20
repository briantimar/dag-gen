""" Models for parameterizing distributions over graphs. """

import torch
from torch.distributions.categorical import Categorical
from .utils import is_valid_adjacency_matrix



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

class BatchDAG:
    """ Container for (a batch of) BatchDAGs that specify computations"""

    def __init__(self, input_dim, output_dim,
                num_intermediate, connections, activations, 
                activation_functions=None, activation_labels=None, check_shapes=True):
        """ `input_dim`: int, the dimensionality of the graph input.
            `output_dim`: int, the dimensionality of the graph output.
            `num_intermediate`: (N,) torch long tensor specifying the number of intermediate vertices of each
            graph in the batch.
            `connections`: (N, max_int + output_dim, max_int + input_dim) uint8 tensor specifying the adjacency
            matrix of each graph in the batch.
            `activations`: (N, max_int + output_dim) int tensor specifying which activation function to apply
            `activation_functions`: if not None, list of shape-preserving functions on torch tensors. 
            at each active vertex. 
            `check_shapes`: Bool, default `True`: whether to check tensor shapes against num_intermediate. 
                Set to `False` if constructing from a subset of the intermediate sizes used to build the `connections`, `activations` tensors.

            Here `N` denotes the batch size of the graph tensors, and `max_int` the largest number of 
            intermediate vertices within the graph batch.

            Both `connections` and `activations` should be left-justified, ie along each dimension the meaningful
            values are packed first. The rest is padding.
            """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_intermediate = num_intermediate
        self.connections = connections
        self.activations = activations

        if activation_functions is not None:
            if activation_labels is not None and len(activation_functions) != len(activation_labels):
                raise ValueError("Must provide exactly one label per activation")
            if activation_labels is None:
                activation_labels = [f"fn{i}" for i in range(len(activation_functions))]
        self.activation_functions = activation_functions
        self.activation_labels = activation_labels
        
        # largest number of intermediate vertices in the batch
        self.max_int = num_intermediate.max().item()
        # largest number of receiving vertices
        self.max_receiving = self.max_int + output_dim
        # largest number of emitting vertices
        self.max_emitting = self.max_int + input_dim
        #largest total graph size
        self.max_size = self.max_receiving + input_dim
        self.batch_size = self.num_intermediate.size(0)

        if check_shapes:
            if tuple(num_intermediate.shape) != (self.batch_size,):
                raise ValueError(f"Invalid num_intermediate shape {num_intermediate.shape}")
            if tuple(connections.shape) != (self.batch_size, self.max_receiving, self.max_emitting):
                print(f"expecting connections shape ({self.batch_size}, {self.max_receiving}, {self.max_emitting})")
                raise ValueError(f"Invalid connections shape {connections.shape}")
            if tuple(activations.shape) != (self.batch_size, self.max_receiving):
                raise ValueError(f"Invalid activations shape {activations.shape}")

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for i in range(self.batch_size):
            yield DAG(self.input_dim, self.output_dim, self.num_intermediate[i], 
                        self.connections[i, ...], self.activations[i, ...],
                        activation_functions=self.activation_functions,
                        activation_labels=self.activation_labels,check_valid=False)

    def set_activation_functions(self, activation_labels):
        """ Sets activation functions to those specified by list `activation_labels` (in that order) """
        from .utils import get_activation
        self.activation_labels = activation_labels
        self.activation_functions = [get_activation(label) for label in activation_labels]

    def _forward_with(self, x, activation_choices):
        """ Compute forward passes for each of the networks on a single input.
            `x`: (M, input_dim) input tensor 
            `activation_choices`: list of candidate activation functions.
            Returns: (M, N,output_dim), obtained by applying the ith network to the ith element of x.
            """
        if len(x.shape) == 1:
            x = x.view(1, -1)
            is_singleton = True
        else:
            is_singleton = False

        if x.size(1) != self.input_dim:
            raise ValueError("Invalid input shape {0} for BatchDAG input size {1}".format(x.shape, self.input_dim))

        data_batch_size = x.size(0)

        # tensor to hold intermediate computation results
        # y[:, :, i] is the graph value at layer i of the topological sort, 
        # or zero where the graph computation has already finished. The first input_dim
        # entries hold the graph input.
        y = torch.zeros(data_batch_size, self.batch_size, self.max_size, dtype=torch.float)
        emitters = y[:, :, :self.max_emitting] 
        #initialize with graph inputs
        y[:, :, :self.input_dim] = x.unsqueeze(1)

        for i in range(self.max_receiving):
            #absolute index of the current vertex
            absolute_index = i + self.input_dim
            #absolute index of the last vertex which could provide input
            largest_input_index = min(self.max_emitting, absolute_index)
            #(N, num_inputs) bool tensor of possible inputs
            input_vertices = self.connections[:, i, :largest_input_index]
            #(M, N, num_inputs) float tensor of existing values in the graph
            input_emitters = emitters[:, :, :largest_input_index]
            #(M, N, num_inputs) tensor of  inputs into the given vertex
            inputs = input_vertices.to(dtype=torch.float).unsqueeze(0) * input_emitters
            #(M, N) tensor of summed inputs at the given vertex
            summed_input = inputs.sum(dim=-1)
            
            #(M, N, NA) tensor of candidate activations
            all_act = torch.stack([f(summed_input) for f in activation_choices], dim=-1)
            # select the correct activation function for each graph
            # (M,N) tensor holding the output of the ith receiving vertex.
            output = all_act[:, range(self.batch_size), self.activations[:, i]]
            #update the current state of the graph with computation results
            y[:, :, absolute_index] = output
            
        #the result of each graph's computation is stored at the vertex index set by its length
        output_start_indices = self.num_intermediate + self.input_dim
        output_end_indices = self.num_intermediate + self.input_dim + self.output_dim
        #(M, N, output_dim) tensor of network outputs
        outputs = [ y[:, i, output_start_indices[i]:output_end_indices[i]] for i in range(self.batch_size)]
        outputs = torch.stack(outputs, dim=1)

        if is_singleton:
            outputs = outputs.view(self.batch_size, self.output_dim)
        return outputs

    def forward(self, x):
        """ Performs forward pass on the inputs x. Requires self.activation_functions to be set. 
            `x`: (data_batch_size, input_size) tensor of inputs.
            Returns: (data_batch_size, dag_batch_size,  output_size) tensor of outputs."""
        if self.activation_functions is None:
            raise ValueError("Activation functions must be set before forward() is called.")
        return self._forward_with(x, self.activation_functions)

    
    def to_graphviz(self):
        """ Returns list of graphviz Digraphs, one for each graph in the batch.
        `activation_labels`: list of strings to label activation functions"""
        return [d.to_graphviz() for d in self]

class DAG(BatchDAG):
    """ For convenience -- represents a single DAG, has no graph batch dimension. """

    def __init__(self,input_dim, output_dim,
                num_intermediate, connections, activations,
                activation_functions = None, activation_labels = None, check_valid=False):
        """ `input_dim`: int, the dimensionality of the graph input.
            `output_dim`: int, the dimensionality of the graph output.
            `num_intermediate`: int specifying the number of intermediate vertices of each
            graph in the batch.
            `connections`: (max_int + output_dim, max_int + input_dim) uint8 tensor specifying the adjacency
            matrix of each graph in the batch.
            `activations`: (max_int + output_dim) int tensor specifying which activation function to apply
            at each active vertex. 
            `activation_functions`: if not None, list of candidate activation functions
            `check_valid`: Bool, default `False`: check whether connections is a valid adjacency matrix.

            Here `max_int` denotes the largest number of intermediate vertices within the graph batch.

            Both `connections` and `activations` should be left-justified, ie along each dimension the meaningful
            values are packed first. The rest is padding.
            """
        if len(connections.shape) < 3:
            connections = connections.unsqueeze(0)
        else:
            if connections.size(0) > 1:
                raise ValueError(f"Invalid connections shape {connections.shape}")
        if len(activations.shape) < 3:
            activations = activations.unsqueeze(0)
        else:
            if activations.size(0) > 1:
                raise ValueError(f"Invalid activations shape {activations.shape}")
        if not isinstance(num_intermediate, torch.Tensor):
            num_intermediate = torch.tensor(num_intermediate, dtype=torch.long).unsqueeze(0)
        else:
            if len(num_intermediate.shape) != 0:
                raise ValueError(f"Invalid num_intermediate shape {num_intermediate.shape}")
            num_intermediate= num_intermediate.view(1)

        if check_valid:
            if not is_valid_adjacency_matrix(connections[0, ...], num_intermediate[0], input_dim, output_dim):
                raise ValueError("connections is not a valid adjacency matrix")

        super().__init__(input_dim, output_dim, num_intermediate, connections, activations, 
                        activation_functions=activation_functions,
                        activation_labels=activation_labels, check_shapes=False)


    def __len__(self):
        raise TypeError

    def _forward_with(self, x, activation_choices):
        """ Compute forward through the dag, using the activation functions provided
        `x`: (M, input_dim) input tensor 
        `activation_choices`: list of candidate activation functions.
        Returns: (M, output_dim), output tensor
        """
        y = super()._forward_with(x, activation_choices)
        if len(x.shape) == 1 :
            return y.view(self.output_dim)
        return y.view(x.size(0), self.output_dim)

    def forward(self, x):
        """ Compute forward through the dag
        `x`: (M, input_dim) input tensor 
        Returns: (M, output_dim), output tensor.
        """
        y = super().forward(x)
        return y

    def sample_action_with_log_prob(self, state):
        """This is implemented only for compatibility with my policy-gradient training code. 
            Given input state (singleton or 1d tensor) return action in [0, ... output_dim) defined by 
            sampling from the categorical distribution which uses DAG logits as final layer. """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_num_intermediate(self):
        """ Returns the number of intermediate vertices in the DAG (int) """
        return self.num_intermediate.item()
    
    @property
    def size(self):
        """ Int, the total number of vertices in the DAG"""
        return self.get_num_intermediate() + self.input_dim + self.output_dim

    def to_graphviz(self):
        """ Returns a graphviz Digraphs
        """
        if self.activation_labels is None:
            raise ValueError("Must specify activation labels before generating graphviz")
        from .utils import build_graphviz
        return build_graphviz(self.input_dim, self.output_dim, self.num_intermediate[0], 
                                self.connections[0, ...], self.activations[0, ...],self.activation_labels)
           

class DAGDistribution(torch.nn.Module):
    """ PyTorch model which defines a distribution over directed acyclic graphs. 
        Should provide:
            * exact sampling from model distribution
            * tractable log-likelihood
        """
    
    def __init__(self):
        super().__init__()
        self.activation_functions = None

    def sample_dags_with_log_probs(self, batch_size, min_intermediate_vertices=None,
                                                    max_intermediate_vertices=None):
        """ Sample a batch of `batch_size` BatchDAGs according to the model distribution.
            `max_intermediate_vertices`: if not None, the max number of non-IO vertices to permit in each graph.
            `min_intermediate_vertices`: if not None, the min number of non-IO vertices to permit in each graph.
        returns: BatchDAG, log_probs
            where BatchDAG has len `batch_size` and `log_probs` is (batch_size,) tensor of corresponding log-probabilities.

        """
        raise NotImplementedError

    def sample_networks_with_log_probs(self, batch_size, min_intermediate_vertices=None,
                                                    max_intermediate_vertices=None):
        """ Sample a batch of `batch_size` networkss according to the model distribution.
            `max_intermediate_vertices`: if not None, the max number of non-IO vertices to permit in each graph.
            `min_intermediate_vertices`: if not None, the min number of non-IO vertices to permit in each graph.
        returns: networks, log_probs
            where networks has len `batch_size` and `log_probs` is (batch_size,) tensor of corresponding log-probabilities.
        Each network should implement a forward() method which accepts batched tensor inputs.
        """
        raise NotImplementedError
       

class GraphRNN(DAGDistribution):
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
    """ Graph generator which uses GRU cells. Each graph sampled from the distribution accepts a single scalar input."""

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

    def _get_samples_and_log_probs(self, logits, mask_to_zero=None):
        """ Obtain samples and log-probabilities, using a Categorical distribution, for the given logits.
            `logits`: a (batch_size, output_dim) float tensor
            Returns : 
                `samples`: (batch_size,) long tensor of samples in the interval [0, output_dim)
                `log_probs`: (batch_size,) float tensor holding the log-probabilities of these samples.
                `mask_to_zero`: if not `None`: (batch_size,) uint8 tensor: elements of `samples` indicated by the mask will be set to zero prior
                 to log-probability computation; elements of log_probs will then be zeroed as well."""
        dist = Categorical(logits=logits)
        samples = dist.sample()
        if mask_to_zero is not None:
            samples[mask_to_zero] = 0
        lps = dist.log_prob(samples)
        if mask_to_zero is not None:
            lps[mask_to_zero] = 0
        return samples, lps

    def _get_log_probs(self, logits, samples, mask_to_zero=None):
        """Compute the log-probability of the samples under a categorical distribution with
        the given logits.
        logits: a (batch_size, output_dim) float Tensor
        samples: (batch_size,) long Tensor with elements taking values in [0, output_dim)
         `mask_to_zero`: if not `None`, (batch_size,) uint8 tensor: elements of `samples` indicated by the mask will be set to zero prior
                 to log-probability computation; elements of log_probs will then be zeroed as well.
        Returns: (batch_size) float tensor holding the log-probabilities.
        """
        if mask_to_zero is not None:
            samples[mask_to_zero] = 0
        dist = Categorical(logits=logits)
        lps = dist.log_prob(samples)
        if mask_to_zero is not None:
            lps[mask_to_zero] = 0
        return lps

    def _sample_graph_tensors_resolved_logprobs(self, N, max_vertices=None, min_vertices=None ):
        """ Sample N graph encodings from the ScalarGraphGRU, with log probs resolved by vertex.
        
        `N`: how many graphs to sample
        `max_vertices`: if not None, the max number of vertices to permit in each graph.
        `min_vertices`: if not None, the sampled graphs will contain at least this many vertices.
        
        Returns: next_active_vertices (N, maxnum) bool tensor specifying which vertices exist in each graph.
                connections_all (N, maxnum, maxnum) bool tensor. The i, j, k element is nonzero iff a connection k -> j exists in the ith graph.
                activations (N, maxnum) int tensor specifying an activation function to be applied at each vertex.
                log_probss: (N, maxnum) tensor of log-probabilities, one for each vertex, conditional on its ancestors."""

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
        
        return next_active_vertices, connections_all, activations, log_probs

    def sample_graph_tensors(self, N, max_vertices=None, min_vertices=None ):
        """ Sample N graph encodings from the ScalarGraphGRU.
        
        `N`: how many graphs to sample
        `max_vertices`: if not None, the max number of vertices to permit in each graph.
        `min_vertices`: if not None, the sampled graphs will contain at least this many vertices.
        
        Returns: next_active_vertices (N, maxnum) bool tensor specifying which vertices exist in each graph.
                connections_all (N, maxnum, maxnum) bool tensor. The i, j, k element is nonzero iff a connection k -> j exists in the ith graph.
                activations (N, maxnum) int tensor specifying an activation function to be applied at each vertex.
                log_probs: (N,) tensor of log-probabilities, one for each sampled graph"""
        next_active_vertices, connections, activations, lps_resolved = self._sample_graph_tensors_resolved_logprobs(N, max_vertices=max_vertices, min_vertices=min_vertices)
        return next_active_vertices, connections, activations, lps_resolved.sum(dim=1)

class GraphGRU(ScalarGraphGRU):
    """ Defines a distribution over graphs which may have arbitrary input / output dimensions."""

    def __init__(self, input_dim, output_dim,
                    hidden_size, logits_hidden_size, num_activations, 
                    min_intermediate_vertices=None, 
                    max_intermediate_vertices=None):
                
        super().__init__(hidden_size, logits_hidden_size, num_activations)
        if input_dim < 1:
            raise ValueError("Invalid input dimension {0}").format(input_dim)
        if output_dim < 1:
            raise ValueError("Invalid output dimension {0}".format(output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_io_vertices = input_dim + output_dim

        self.min_intermediate_vertices = min_intermediate_vertices
        self.max_intermediate_vertices = max_intermediate_vertices

    def set_activation_functions(self, activation_labels):
        """Sets activation functions according to the list of labels provided (which should match those expected by utils.get_activation)
        """
        from .utils import get_activation
        self.activation_labels = activation_labels
        self.activation_functions = [get_activation(f) for f in activation_labels]

    def _sample_graph_tensors_resolved(self, N, max_intermediate_vertices=None, min_intermediate_vertices=None): 
        """ Sample N graph encodings from the GraphGRU with log probs.
        
        `N`: how many graphs to sample
        `max_intermediate_vertices`: if not None, the max number of non-IO vertices to permit in each graph.
        `min_intermediate_vertices`: if not None, the min number of non-IO vertices to permit in each graph.
        
        Let maxnum denote the max number of intermediate vertices among all sampled graphs. Then return values are
                `num_intermediate` (N,) integer tensor specifying the number of intermediate vertices in each sampled graph.
                `connections` maxnum+output_dim length of byte tensors. 
                    The ith tensor has shape (N, num_prev_emitting), where num_prev_emitting is the number of emitting vertices
                    that come before receiving vertex i in the graph. 
                    connections[i][:,j] is a byte tensor indicating, for each graph in the batch, whether a connection j -> i exists.
                `activations` maxnum+output_dim length list of integer tensors. Each specifies the activation applied at a particular vertex in each graph; 
                     The last output_dim entries always exist and correspond to the activations applied to the output neurons.
                `log_probs`: maxnum + output_dim length list of log-probabilities.
        
        Vertices are ordered in the following manner:
            the first input_dim are the input vertices
            the last output_dim are the output vertices
            All vertices in between, if any, are the intermediate vertices.
            """

        if max_intermediate_vertices is not None and min_intermediate_vertices is not None and (max_intermediate_vertices < min_intermediate_vertices):
            raise ValueError("Invalid vertex number specs: min, max = {0}, {1}".format(min_intermediate_vertices, max_intermediate_vertices))
        if max_intermediate_vertices is not None and max_intermediate_vertices < 1:
            raise ValueError("Invalid max vertex number {0}".format(max_intermediate_vertices))

        #index of the current vertex
        # start sampling after the input vertices
        vertex_index = self.input_dim

        graph_complete = False
        vertex_hidden_state = self.hidden_init.view(1, -1).expand(N, self.hidden_size)
        vertex_input = self.vertex_input_init.view(1, -1).expand(N, self.vertex_input_size)

        # indicates the number of intermediate vertices in each sampled graph
        num_intermediate = torch.zeros(N, dtype=torch.long)

        #holds log probabilities for the batch
        log_probs = []
        # holds connectivity matrices
        all_connections = []
        #holds sampled activation functions
        activations = []

        #keep track of which graphs have more neurons to sample.
        active_graphs = torch.ones(N,dtype=torch.uint8)
        
        while not graph_complete:
            #compute new hidden state
            vertex_hidden_state = self.vertex_cell(vertex_input, vertex_hidden_state)
            #check whether another vertex should be added
            vertex_logits = self.vertex_logits(vertex_hidden_state)
        
            add_vertex, lp_vertex = self._get_samples_and_log_probs(vertex_logits)
            
            #a graph is active only if it has added a vertex at each sampling step.
            if min_intermediate_vertices is not None and (vertex_index+1 - self.input_dim) <= min_intermediate_vertices:
                add_vertex = torch.ones(N, dtype=torch.long)
                lp_vertex = torch.zeros(N, requires_grad=True)
                min_vertices_satisfied = False
            else:
                min_vertices_satisfied = True

            if max_intermediate_vertices is not None and (vertex_index+1 - self.input_dim) > max_intermediate_vertices:
                add_vertex = torch.zeros(N, dtype=torch.long)
                lp_vertex = torch.zeros(N, requires_grad=True)
                max_vertices_satisfied = True
            else:
                max_vertices_satisfied = False

            active_graphs = active_graphs & (add_vertex > 0)
            #ignore graphs which have already finished sampling
            mask_to_zero = ~active_graphs
            num_intermediate += active_graphs.to(dtype=torch.long)

            ##sampling halts when all graph sequences have terminated, or max number of vertices is reached
            graph_complete = min_vertices_satisfied and ( active_graphs.sum().item()==0 or max_vertices_satisfied)


            if not graph_complete:
                #now pass hidden state down to edge cells
                edge_hidden_state = vertex_hidden_state
                edge_input_init = torch.zeros(N,1)
                edge_input = edge_input_init

                #determine whether to connect to each previous vertex
                vertex_connections = []
                for prev_index in range(vertex_index):
                    #compute edge hidden state
                    edge_hidden_state = self.edge_cell(edge_input, edge_hidden_state)            
                    #sample connectivity to prev_index
                    connection_logits = self.edge_logits(edge_hidden_state)

                    add_connection, lp_connection = self._get_samples_and_log_probs(connection_logits, mask_to_zero=mask_to_zero)
                                        
                    lp_vertex = lp_vertex + lp_connection
                    vertex_connections.append(add_connection)

                    edge_input = add_connection.view(N, 1).to(dtype=torch.float)

                all_connections.append(torch.stack(vertex_connections, dim=1))
                # finally, determine which activation function to apply to this vertex
                act_state = self.activation_cell(edge_input, edge_hidden_state)
                act_logits = self.activation_logits(act_state)

                activation, lp_act = self._get_samples_and_log_probs(act_logits, mask_to_zero=mask_to_zero)
                lp_vertex = lp_vertex + lp_act

                activations.append(activation)
                #record the conditional log-probability for everything sampled at this vertex
                log_probs.append(lp_vertex)

                #compute input to the next vertex cell
                vertex_input = torch.cat((edge_hidden_state, activation.view(-1, 1).to(dtype=torch.float)), dim=1)
                
                vertex_index += 1

            else:
                log_probs.append(lp_vertex)

        #at this point, there are no more intermediate vertices to sample
        #however, the connections to the output vertices still have to be defined
        maxnum = vertex_index - self.input_dim
        max_graph_size = maxnum + self.input_dim + self.output_dim
        # the outputs are treated as vertices which exist with probability 1, but can't connect to each other.
        # NOTE activations are applied to the output vertices!
        if num_intermediate.sum().item()==0:
            edge_hidden_state = vertex_hidden_state
        for vertex_index in range(vertex_index, max_graph_size):
            edge_input_init = torch.zeros(N,1)
            edge_input = edge_input_init
            lp_vertex = torch.zeros(N)

            vertex_connections = []

            for prev_index in range(maxnum + self.input_dim):
                #compute edge hidden state
                edge_hidden_state = self.edge_cell(edge_input, edge_hidden_state)            
                #sample connectivity to prev_index
                connection_logits = self.edge_logits(edge_hidden_state)
                # for each graph, only look at prev connections which are actually possible for that graph.
                vertex_exists = (num_intermediate + self.input_dim) > prev_index
                mask_to_zero = ~vertex_exists
                add_connection, lp_connection = self._get_samples_and_log_probs(connection_logits, mask_to_zero=mask_to_zero)

                lp_vertex[vertex_exists] = lp_connection[vertex_exists] + lp_connection[vertex_exists]
                vertex_connections.append(add_connection)

                edge_input = add_connection.view(N, 1).to(dtype=torch.float)
           
            all_connections.append(torch.stack(vertex_connections, dim=1))
            
            act_state = self.activation_cell(edge_input, edge_hidden_state)
            act_logits = self.activation_logits(act_state)
            activation, lp_act = self._get_samples_and_log_probs(act_logits)
            lp_vertex = lp_vertex + lp_act

            activations.append(activation)
            log_probs.append(lp_vertex)

            edge_hidden_state = act_state

        #return the lists of tensors which together define graphs

        return num_intermediate, all_connections, activations, log_probs


    def sample_graph_tensors(self, N, max_intermediate_vertices=None, min_intermediate_vertices=None): 
        """ Sample N graph encodings from the GraphGRU with log probs.
        
        `N`: how many graphs to sample
        `max_intermediate_vertices`: if not None, the max number of non-IO vertices to permit in each graph.
        `min_intermediate_vertices`: if not None, the min number of non-IO vertices to permit in each graph.
        
        Let maxnum denote the max number of intermediate vertices among all sampled graphs. Then return values are
                `num_intermediate` (N,) integer tensor specifying the number of intermediate vertices in each sampled graph.
                `connections` (N, maxnum + output_dim, maxnum + input_dim) byte tensor. 
                    The i, j, k element is nonzero iff a connection k + input_dim -> j + output_dim exists in the ith graph.
                `activations` (N, maxnum + output_dim) int tensor specifying an activation function to be applied at each intermediate vertex.
                    
                    Takes values in 0, ... num_activations -1
                `log_probs`: (N,) float tensor of log-probabilities assigned to each graph in the batch
        
        Generally each sampled graph will have less than maxnum intermediate vertices.
         In these cases, only certain portions of each slice of the returned tensors
        are actually used to specify a graph. Suppose the ith graph has n intermediate vertices. Then:
            Its activations are specified in activations[i, :(n+output_dim)]
            Its connections are specified in connections[i, :(n + output_dim), :(n + input_dim)]
        The other entries of these tensors (for a particular batch index) should be ignored.

        Vertices are ordered in the following manner:
            the first input_dim are the input vertices
            the last output_dim are the output vertices
            All vertices in between, if any, are the intermediate vertices.
            """
        from torch.nn.utils.rnn import pad_sequence
        
        num_intermediate, connections_by_vertex, activations_by_vertex, log_probs_by_vertex = self._sample_graph_tensors_resolved(N, max_intermediate_vertices=max_intermediate_vertices, 
                                                                                                        min_intermediate_vertices=min_intermediate_vertices)
        max_num_intermediate = num_intermediate.max().item()
        #the number of 'active' neurons, which have activations and have nontrivial connections (everything but the inputs)
        max_num_active = max_num_intermediate + self.output_dim
        assert len(connections_by_vertex) == max_num_active
        assert len(activations_by_vertex) == max_num_active
        # the extra entry here comes from the probability of the 'stop' token 
        assert len(log_probs_by_vertex) == max_num_active + 1
       
        #first, stack the activations together
        activations = torch.stack(activations_by_vertex, dim=1)
        for i in range(N):
            ni = num_intermediate[i]
            activations[i, ni:ni+self.output_dim] = activations[i, -self.output_dim:]
            activations[i, ni+self.output_dim:] = -1

            
        connections = pad_sequence([c.permute(1, 0) for c in connections_by_vertex], batch_first=True)
        connections = connections.permute(2, 0, 1)
        #finally, move the nonzero entries around so that they all come first for a given batch index
        for i in range(N):
            ni = num_intermediate[i]
            connections[i, ni:ni+self.output_dim, ...] = connections[i,-self.output_dim:, ... ]
            connections[i, ni+self.output_dim:, ...] = 0 
        
        log_probs = torch.stack(log_probs_by_vertex, dim=1).sum(dim=1)

        return num_intermediate, activations, connections, log_probs

    def sample_dags_with_log_probs(self, batch_size ):
        """ Sample a batch of `batch_size` BatchDAGs according to the GraphGRU's distribution.
      """
        num_intermediate, activations, connections, log_probs = self.sample_graph_tensors(batch_size, 
                                                    max_intermediate_vertices=self.max_intermediate_vertices,
                                                    min_intermediate_vertices=self.min_intermediate_vertices)
        batched_dag = BatchDAG(self.input_dim, self.output_dim, 
                            num_intermediate, connections, activations )
        return batched_dag, log_probs

    def sample_networks_with_log_probs(self, batch_size):
        """ Sample a batch of `batch_size` networks according to the model distribution.
        
        returns: networks, log_probs
            where networks has len `batch_size` and `log_probs` is (batch_size,) tensor of corresponding log-probabilities.
        """
        if self.activation_functions is None:
            raise ValueError("Activation functions must be selected before sampling networks.")
        batchdag, log_probs = self.sample_dags_with_log_probs(batch_size)
        batchdag.activation_functions = self.activation_functions
        if hasattr(self, 'activation_labels'):
            batchdag.activation_labels = self.activation_labels
        return [dag for dag in batchdag], log_probs
    
    def _log_probs_from_resolved_tensors(self, num_intermediate, connections, activations): 
        """ Compute log probabilities of the graph tensors provided. 

            `num_intermediate` (N,) integer tensor specifying the number of intermediate vertices in each graph.
            `connections` maxnum+output_dim length of byte tensors. 
                    The ith tensor has shape (N, num_prev_emitting), where num_prev_emitting is the number of emitting vertices
                    that come before receiving vertex i in the graph. 
                    connections[i][:,j] is a byte tensor indicating, for each graph in the batch, whether a connection j -> i exists.
            `activations` maxnum+output_dim length list of (N,) integer tensors. Each specifies the activation applied at a particular vertex in each graph; 
                    The last output_dim entries always exist and correspond to the activations applied to the output neurons.
            `log_probs`: maxnum + output_dim length list of log-probabilities.

        Vertices are ordered in the following manner:
            the first input_dim are the input vertices
            the last output_dim are the output vertices
            All vertices in between, if any, are the intermediate vertices.
            """

        #index of the current vertex
        # start sampling after the input vertices
        vertex_index = self.input_dim

        # batch size
        N = num_intermediate.size(0)

        graph_complete = False
        #initialize the hidden state
        vertex_hidden_state = self.hidden_init.view(1, -1).expand(N, self.hidden_size)
        vertex_input = self.vertex_input_init.view(1, -1).expand(N, self.vertex_input_size)

        #holds log probabilities for the batch
        log_probs = []

        #keep track of which graphs are still being evaluated
        active_graphs = torch.ones(N,dtype=torch.uint8)
        
        while not graph_complete:
            #compute new hidden state
            vertex_hidden_state = self.vertex_cell(vertex_input, vertex_hidden_state)
            #check whether another vertex should be added
            vertex_logits = self.vertex_logits(vertex_hidden_state)
        
            #whether each graph requires another vertex to be added
            add_vertex = (num_intermediate > (vertex_index - self.input_dim)).to(dtype=torch.long)

            lp_vertex = self._get_log_probs(vertex_logits, add_vertex)
            
            active_graphs = active_graphs & (add_vertex > 0)
            
            #ignore graphs which have already finished sampling
            mask_to_zero = ~active_graphs
        
            ##intermediate eval halts when no graphs are active
            graph_complete = ( active_graphs.sum().item()==0 )

            if not graph_complete:
                #now pass hidden state down to edge cells
                edge_hidden_state = vertex_hidden_state
                edge_input_init = torch.zeros(N,1)
                edge_input = edge_input_init

                #connections made at this vertex for each graph
                vertex_connections = connections[vertex_index-self.input_dim]
                for prev_index in range(vertex_index):
                    #compute edge hidden state
                    edge_hidden_state = self.edge_cell(edge_input, edge_hidden_state)            
                    
                    #likelihood of connection to prev_index
                    connection_logits = self.edge_logits(edge_hidden_state)
                    prev_conn = vertex_connections[:, prev_index]
                    lp_connection = self._get_log_probs(connection_logits, prev_conn, mask_to_zero=mask_to_zero)
                                        
                    lp_vertex = lp_vertex + lp_connection
                    
                    edge_input = prev_conn.view(N, 1).to(dtype=torch.float)

                # now compute the log-probability of activation
                activation = activations[vertex_index - self.input_dim]
                act_state = self.activation_cell(edge_input, edge_hidden_state)
                act_logits = self.activation_logits(act_state)

                lp_act = self._get_log_probs(act_logits, activation, mask_to_zero=mask_to_zero)
                lp_vertex = lp_vertex + lp_act

                #record the conditional log-probability for everything sampled at this vertex
                log_probs.append(lp_vertex)

                #compute input to the next vertex cell
                vertex_input = torch.cat((edge_hidden_state, activation.view(-1, 1).to(dtype=torch.float)), dim=1)
                
                vertex_index += 1

            else:
                #record log-prob corresponding to the 'stop' token
                log_probs.append(lp_vertex)
            

        #at this point, there are no more intermediate vertices to sample
        #however, the connections to the output vertices still have to be defined
        maxnum = vertex_index - self.input_dim
        max_graph_size = maxnum + self.input_dim + self.output_dim
        # the outputs are treated as vertices which exist with probability 1, but can't connect to each other.
        # NOTE activations are applied to the output vertices!
        if num_intermediate.sum().item()==0:
            edge_hidden_state = vertex_hidden_state
        for vertex_index in range(vertex_index, max_graph_size):
            edge_input_init = torch.zeros(N,1)
            edge_input = edge_input_init
            lp_vertex = torch.zeros(N)

            vertex_connections = connections[vertex_index - self.input_dim]

            for prev_index in range(maxnum + self.input_dim):
                #compute edge hidden state
                edge_hidden_state = self.edge_cell(edge_input, edge_hidden_state)            
                #sample connectivity to prev_index
                connection_logits = self.edge_logits(edge_hidden_state)
                                
                prev_conn = vertex_connections[:, prev_index]
                # check that the requested connections are actually possible
                vertex_exists = (num_intermediate + self.input_dim) > prev_index
                mask_to_zero = ~vertex_exists
                if (prev_conn[mask_to_zero]>0).any():
                    raise ValueError("Graph tensors are invalid!")

                lp_connection = self._get_log_probs(connection_logits, prev_conn, mask_to_zero=mask_to_zero)

                lp_vertex[vertex_exists] = lp_connection[vertex_exists] + lp_connection[vertex_exists]
                edge_input = prev_conn.view(N, 1).to(dtype=torch.float)
            
            
            act_state = self.activation_cell(edge_input, edge_hidden_state)
            act_logits = self.activation_logits(act_state)
            activation = activations[vertex_index - self.input_dim]
            lp_act = self._get_log_probs(act_logits, activation)
            lp_vertex = lp_vertex + lp_act

            log_probs.append(lp_vertex)

            edge_hidden_state = act_state

        #return the lists of tensors which together define graphs

        return sum(log_probs)

    def log_probs_from_tensors(self, num_intermediate, connections_left_justified, activations_left_justified):
        """Compute the log-probabilities of the graphs defined by the given connectivity and activation tensors.
            N= batch size of the inputs.
            num_intermediate: (N,) long tensor specifying the number of intermediate vertices in each graph.
            connections: (N, num_receiving, num_emitting) byte tensor
                i, j, k is nonzero iff k --> input_dim + j in graph i
            activations: (N, num_receiving) long tensor specifying activation values.
            where num_receiving = max(num_intermediate) + output_dim
                num_emitting = max(num_intermediate) + input_dim
            Returns: (N,) tensor of log-probabilities

            TODO don't call the resolved version, there's no need.
            """
        from .utils import to_resolved_tensors
        connections_resolved, activations_resolved = to_resolved_tensors(num_intermediate, connections_left_justified, activations_left_justified, 
                                                                        self.input_dim, self.output_dim)
        return self._log_probs_from_resolved_tensors(num_intermediate, connections_resolved, activations_resolved)

    def log_probs_from_dag(self, dag):
        """ Compute the log-probability of the given BatchDAG under the current model.
            Returns: tensor of length len(dag) holding log-probabilities. """
        return self.log_probs_from_tensors(dag.num_intermediate, dag.connections, dag.activations)