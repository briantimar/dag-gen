""" Models for parameterizing distributions over graphs. """

import torch
from torch.distributions.categorical import Categorical

class ScalarTorchDag:
    """ Define and apply batched computational graphs based on torch mask tensors and a list of activation functions.
        Only allows for scalar inputs and outputs."""

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

    def to_graphviz(self):
        """Return graphvis objects representing each graph in the batch"""
        pass

            

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

class DAG:
    """ Container for (a batch of) DAGs that specify computations"""

    def __init__(self, input_dim, output_dim,
                num_intermediate, connections, activations):
        """ `input_dim`: int, the dimensionality of the graph input.
            `output_dim`: int, the dimensionality of the graph output.
            `num_intermediate`: (N,) torch long tensor specifying the number of intermediate vertices of each
            graph in the batch.
            `connections`: (N, max_int + output_dim, max_int + input_dim) uint8 tensor specifying the adjacency
            matrix of each graph in the batch.
            `activations`: (N, max_int + output_dim) int tensor specifying which activation function to apply
            at each active vertex. 

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

        # largest number of intermediate vertices in the batch
        self.max_int = num_intermediate.max().item()
        # largest number of receiving vertices
        self.max_receiving = self.max_int + output_dim
        # largest number of emitting vertices
        self.max_emitting = self.max_int + input_dim
        #largest total graph size
        self.max_size = self.max_receiving + input_dim
        self.batch_size = self.num_intermediate.size(0)

        if tuple(num_intermediate.shape) != (self.batch_size,):
            raise ValueError(f"Invalid num_intermediate shape {num_intermediate.shape}")
        if tuple(connections.shape) != (self.batch_size, self.max_receiving, self.max_emitting):
            raise ValueError(f"Invalid connections shape {connections.shape}")
        if tuple(activations.shape) != (self.batch_size, self.max_receiving):
            raise ValueError(f"Invalid activations shape {activations.shape}")

    def forward(self, x, activation_choices):
        """ Compute forward passes for each of the networks on a single input.
            `x`: (input_dim,) input tensor 
            `activation_choices`: list of candidate activation functions.
            Returns: (N,output_dim), obtained by applying the ith network to the ith element of x.
            """
        if tuple(x.shape) != (self.input_dim,):
            raise ValueError("Invalid input shape {0} for DAG input size {1}".format(x.shape, self.input_dim))

        # tensor to hold intermediate computation results
        # y[:, i] is the graph value at layer i of the topological sort, 
        # or zero where the graph computation has already finished. The first input_dim
        # entries hold the graph input.
        y = torch.zeros(self.batch_size, self.max_size, dtype=torch.float)
        emitters = y[:, :self.max_emitting]
        #initialize with graph inputs
        y[:, :self.input_dim] = x

        for i in range(self.max_receiving):
            #which vertices provide inputs to the current vertex
            #(N, max_emitting)
            input_vertices = self.connections[:, i, :]
            #(N,) tensor of summed inputs into the given vertex
            inputs = input_vertices.to(dtype=torch.float) * emitters
            summed_input = inputs.sum(dim=1)
            
            #(N, NA) tensor of candidate activations
            all_act = torch.stack([f(summed_input) for f in activation_choices], dim=1)
            # select the correct activation function for each graph
            # (N,) tensor holding the output of the ith receiving vertex.
            output = all_act[range(self.batch_size), self.activations[:, i]]
            #update the current state of the graph with computation results
            y[:, i+self.input_dim] = output
            
        #the result of each graph's computation is stored at the vertex index set by its length
        output_start_indices = self.num_intermediate + self.input_dim
        output_end_indices = self.num_intermediate + self.input_dim + self.output_dim
        outputs = [ y[i, output_start_indices[i]:output_end_indices[i]] for i in range(self.batch_size)]
        return torch.stack(outputs, dim=0)
    

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

    def _get_samples_and_log_probs(self, logits):
        """ Obtain samples and log-probabilities, using a Categorical distribution, for the given logits."""
        dist = Categorical(logits=logits)
        samples = dist.sample()
        lps = dist.log_prob(samples)
        return samples, lps

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
                    hidden_size, logits_hidden_size, num_activations ):
        super().__init__(hidden_size, logits_hidden_size, num_activations)
        if input_dim < 1:
            raise ValueError("Invalid input dimension {0}").format(input_dim)
        if output_dim < 1:
            raise ValueError("Invalid output dimension {0}".format(output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_io_vertices = input_dim + output_dim

    def _sample_graph_tensors_resolved(self, N, max_intermediate_vertices=None, min_intermediate_vertices=None): 
        """ Sample N graph encodings from the GraphGRU with log probs.
        
        `N`: how many graphs to sample
        `max_intermediate_vertices`: if not None, the max number of non-IO vertices to permit in each graph.
        `min_intermediate_vertices`: if not None, the min number of non-IO vertices to permit in each graph.
        
        Let maxnum denote the max number of intermediate vertices among all sampled graphs. Then return values are
                `num_intermediate` (N,) integer tensor specifying the number of intermediate vertices in each sampled graph.
                `connections` maxnum+output_dim length list of byte tensor lists. Each inner list has length maxnum + input_dim.
                    connections[i][j] is a byte tensor indicating, for each graph in the batch, whether a connection j -> i exists.
                `activations` maxnum+output_dim length list of integer tensors. Each specifies the activation applied at a particular vertex in each graph; entries are -1 if the vertex 
                does not exist in that graph. The last output_dim entries always exist and correspond to the activations applied to the output neurons.
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
                lp_vertex = torch.zeros(N)
                min_vertices_satisfied = False
            else:
                min_vertices_satisfied = True

            if max_intermediate_vertices is not None and (vertex_index+1 - self.input_dim) > max_intermediate_vertices:
                add_vertex = torch.zeros(N, dtype=torch.long)
                lp_vertex = torch.zeros(N)
                max_vertices_satisfied = True
            else:
                max_vertices_satisfied = False

            active_graphs = active_graphs & (add_vertex > 0)
            
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
                    add_connection, lp_connection = self._get_samples_and_log_probs(connection_logits)
                    #ignore graphs which have already finished sampling
                    add_connection[~active_graphs] = 0
                    lp_connection[~active_graphs] = 0

                    lp_vertex += lp_connection
                    vertex_connections.append(add_connection)

                    edge_input = add_connection.view(N, 1).to(dtype=torch.float)
                
                all_connections.append(torch.stack(vertex_connections, dim=1))
                # finally, determine which activation function to apply to this vertex
                act_state = self.activation_cell(edge_input, edge_hidden_state)
                act_logits = self.activation_logits(act_state)
                activation, lp_act = self._get_samples_and_log_probs(act_logits)
                activation[~active_graphs] = -1
                lp_act[~active_graphs] = 0
                lp_vertex += lp_act

                activations.append(activation)
                #record the conditional log-probability for everything sampled at this vertex
                log_probs.append(lp_vertex)

                #compute input to the next vertex cell
                vertex_input = torch.cat((edge_hidden_state, activation.view(-1, 1).to(dtype=torch.float)), dim=1)
                
                vertex_index += 1

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
                add_connection, lp_connection = self._get_samples_and_log_probs(connection_logits)
                # for each graph, only look at prev connections which are actually possible for that graph.
                vertex_exists = (num_intermediate + self.input_dim) > prev_index
                add_connection[~vertex_exists] = 0
                lp_vertex[vertex_exists] += lp_connection[vertex_exists]
                vertex_connections.append(add_connection)

                edge_input = add_connection.view(N, 1).to(dtype=torch.float)
           
            all_connections.append(torch.stack(vertex_connections, dim=1))
            
            act_state = self.activation_cell(edge_input, edge_hidden_state)
            act_logits = self.activation_logits(act_state)
            activation, lp_act = self._get_samples_and_log_probs(act_logits)
            lp_vertex += lp_act

            activations.append(activation)
            log_probs.append(lp_vertex)
        
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
                    Undefined entries (corresponding to vertices not present in sampled graphs) are filled with -1
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
        assert len(log_probs_by_vertex) == max_num_active
       
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