import graphviz
import torch

_ACTIVATIONS = {'id': lambda x : x, 
                    'inv': lambda x : -x, 
                    'abs': lambda x : x.abs(), 
                    'relu': lambda x : x.relu(), 
                    'sin': lambda x : x.sin(), 
                    'cos': lambda x : x.cos(), 
                    'gauss': lambda x : (-x**2).exp(),
                    'bias1': lambda x : torch.ones_like(x),
                    }
    
def get_activation(name):
    """Returns one of several standard activation functions."""
    if name not in _ACTIVATIONS.keys():
        raise ValueError(f"{name} is not a valid activation function.")
    return _ACTIVATIONS[name]

def right_justify(arr, lengths, output_dim):
    """ Return a right-justified version of the given array along dim 1.
    arr: (N, k, ...) tensor
    lengths: (N,) list of integers, which specify how much of each row of arr is to be shifted
    In each row, elements lengths[i]: lengths[i] + output_dim are slid all the way to the right.
    """
    justified = torch.zeros_like(arr)
    for i in range(arr.shape[0]):
        justified[i, -(output_dim):, ...] = arr[i, lengths[i]:lengths[i] + output_dim, ...]
        justified[i, :lengths[i], ...] = arr[i, :lengths[i], ...]
    return justified

def dag_dataset_from_batchdag(batchdag):
    """ Constructs a dataset of DAGs from a single BatchDAG. """
    from .models import DAGDataset
    return DAGDataset([d for d in batchdag])

def dag_dataset_from_tensors(input_dim, output_dim,
                num_intermediate, connections, activations, 
                activation_functions=None, activation_labels=None, check_shapes=True):
    """ Creates a dataset of DAGS based on the given batched connectivity and activation tensors."""
    from .models import BatchDAG
    bd = BatchDAG(input_dim, output_dim, num_intermediate, connections, activations, 
                    activation_functions=activation_functions, activation_labels=activation_labels, check_shapes=check_shapes)
    return dag_dataset_from_batchdag(bd)

def collate_dags(dags):
    """ Collates a list of dags into a single BatchDAG."""
    from .models import BatchDAG
    input_dim = dags[0].input_dim
    output_dim = dags[0].output_dim
    activation_functions = dags[0].activation_functions
    activation_labels = dags[0].activation_labels

    connections_all = [dag.connections for dag in dags]
    activations_all = [dag.activations for dag in dags]
    num_intermediate_all = [dag.num_intermediate for dag in dags]

    connections = torch.cat(connections_all, dim=0)
    activations = torch.cat(activations_all, dim=0)
    num_intermediate = torch.cat(num_intermediate_all, dim=0)
    

    return BatchDAG(input_dim, output_dim, num_intermediate, connections, activations, 
                    activation_functions=activation_functions, activation_labels=activation_labels, check_shapes=True)


def to_resolved_tensors( num_intermediate, connections_left_justified, activations_left_justified, input_dim, output_dim):
    """Convert the given monolithic torch tensors defining a batch of graphs to lists of resolved tensors 
        num_intermediate: (N,) long tensor specifying the number of intermediate vertices in each graph.
        connections_left_justified: (N, num_receiving, num_emitting) byte tensor
            i, j, k is nonzero iff k --> input_dim + j in graph i
        activations_left_justified: (N, num_receiving) long tensor specifying activation values.
        where num_receiving = max(num_intermediate) + output_dim
            num_emitting = max(num_intermediate) + input_dim
        Returns: num_intermediate, connections_resolved, activations_resolved.

        """
    from .utils import right_justify
    num_rec = connections_left_justified.size(1)
    num_emit = connections_left_justified.size(2)
    
    activations = right_justify(activations_left_justified, num_intermediate, output_dim)
    connections = right_justify(connections_left_justified, num_intermediate, output_dim)

    conns_resolved = []
    activations_resolved = []
    for i in range(num_rec):
        numinputs = min(input_dim + i, num_emit)
        #split up tensors to avoid issues with in-place mods
        conns = torch.zeros_like(connections[:, i, :numinputs])
        conns.copy_(connections[:, i, :numinputs])

        acts = torch.zeros_like(activations[:,i])
        acts.copy_(activations[:,i])

        conns_resolved.append(conns)
        activations_resolved.append(acts)

    return conns_resolved, activations_resolved

def is_valid_adjacency_matrix(connections, num_intermediate, num_input, num_output):
    """ Check whether input defines a valid, left-justified adjacency matrix.
    `connections`: (max_num_receiving, max_num_emitting) uint8 tensor
    `num_intermediate`: int, number of intermediate vertices
    `num_input`: int, number of input vertices
    `num_output`: int, number of output vertices"""

    num_emitting = num_intermediate + num_input
    num_receiving = num_intermediate + num_output

    if connections.size(0) < num_receiving:
        return False
    if connections.size(1) < num_emitting:
        return False

    embedded_intermediate_size = connections.size(0) - num_output
    #check that dimensions of the connectivity tensor are consistent with single fixed intermediate size
    if embedded_intermediate_size < 0 or embedded_intermediate_size != connections.size(1) - num_input:
        return False

    # check left-justified
    if connections[num_receiving:, :].sum().item() > 0:
        return False
    if connections[:, num_emitting:].sum().item() > 0:
        return False
    # check that vertices only receive input from ancestors
    for i in range(num_receiving):
        if connections[i, i+ num_input:].sum().item() > 0:
            return False
    return True


def build_graphviz(input_dim, output_dim, num_intermediate, 
                connections, activations, activation_labels):
    """ Build graphviz object corresponding to the single DAG specified by the input tensors.
        `input_dim`: int, number of input vertices
        `output_dim`: int, number of output vertices
        `num_intermediate`: int, number of intermediate vertices
        `connections`: (max_num_receiving, max_num_emitting) uint8 adjacency matrix
        `activations`: (max_num_receiving,) int tensor specifying activations to apply at each
        receiving vertex
        `activation_labels`: list of activation function labels. 
        
        Returns: graphviz digraph"""
    
    if not is_valid_adjacency_matrix(connections, num_intermediate, input_dim, output_dim):
        raise ValueError("Connectivity matrix is invalid")
    num_emitting = num_intermediate + input_dim
    num_receiving = num_intermediate + output_dim
    size = num_emitting + output_dim
    dag = graphviz.Digraph()
    #add nodes labeled by activation functions
    for i in range(size):
        node=str(i)
        if i < input_dim:
            label = 'inp%d' % i
        else:
            label = activation_labels[activations[i-input_dim]]
            if i >= num_emitting:
                label = f"out{i-num_emitting};{label}"
        dag.node(node, label=label)
    #add edges
    edgelist = []
    for i in range(num_receiving):
        rec_index = i + input_dim
        for emitting_index in range(min(rec_index, num_emitting)):
            if connections[i, emitting_index] > 0:
                edgelist.append((str(emitting_index), str(rec_index)))
    dag.edges(edgelist)
    return dag

#utilites for constructing graphs
def build_empty_graph(input_dim, output_dim, num_intermediate):
    """ Returns a graph with no connections and trivial activations. """
    from .models import DAG
    num_emit, num_rec = num_intermediate + input_dim, num_intermediate + output_dim
    activations = torch.zeros(num_rec, dtype=torch.long)
    connections = torch.zeros(num_rec, num_emit, dtype=torch.long)

    return DAG(input_dim, output_dim, num_intermediate, connections, activations, check_valid=True)

##### TRAINING ROUTINES

def do_score_training(dag_model, score_function, 
                        total_samples, batch_size, optimizer, 
                            score_logger=None,
                            entropy_logger=None,
                            cost_logger = None,
                            reward_cost_logger=None, 
                            entropy_cost_logger=None,
                            network_callbacks=[],
                            log_every=1,
                            baseline='running_average', 
                            entropic_penalty=0.0):
    """
    Run score-based "policy-gradient" training on the given DAG model.
    `dag_model`: A generative model over DAGs which produces DAG objects and associated log-probability tensors
        should implement `sample_networks_with_log_probs()`
    `score_function`: a function which takes a `DAG` as input and returns a scalar (float) score.
    `total_samples`: int, total number of samples to be drawn from DAG model during training.
    `batch_size`: int, how many samples to use per update step
    `optimizer`: a torch.optim optimizer for the DAG model parameters. 

    `score_logger`: if not None, callback to be called on each batch score value
    `network_callbacks`: list of functions, each of which will be applied to the batch of
    dags sampled from the network at each logging step.
    `baseline`: whether and how to subtract baseline from the score functions.
        choices: ("running_average", `None`)
    `log_every`: Number of update steps between logging.
    `entropic penalty`: hyperparameter in the cost function used to encourage high entropy in the DAG distribution.
    Training attempts to maximize the expected score via gradient descent. """

    if baseline not in ("running_average", None):
        raise ValueError(f"Invalid baseline method {baseline}")
    
    from math import ceil
    #total number of update steps
    num_update = ceil(total_samples / batch_size)

    def get_batch_size(update_index):
        if update_index < num_update -1:
            return batch_size
        rem = total_samples % batch_size
        if rem == 0:
            return batch_size
        return rem

    running_avg_score = 0.

    def cost_function(scores, log_probs):
        if baseline == "running_average":
            bl = running_avg_score
        elif baseline is None:
            bl = 0
        reward_cost = - ((scores - bl) * log_probs).sum() 
        entropy_cost = entropic_penalty * log_probs.mean()
        return reward_cost, entropy_cost

    batch_scores = []

    for update_index in range(num_update):
        # generate samples
        _batch_size = get_batch_size(update_index)
        dags, log_probs = dag_model.sample_networks_with_log_probs(_batch_size)
        scores = torch.tensor(list(map(score_function, dags)))

        reward_cost, entropy_cost = cost_function(scores, log_probs)
        cost = reward_cost + entropy_cost
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        batch_entropy = -log_probs.mean().item()
        batch_score = scores.mean().item()
        batch_scores.append(batch_score)
        running_avg_score = .9 * running_avg_score + .1 * batch_score

        if update_index % log_every == 0:
            if score_logger is not None:
                score_logger(batch_score)
            if entropy_logger is not None:
                entropy_logger(batch_entropy)
            if cost_logger is not None:
                cost_logger(cost.item())
            if reward_cost_logger is not None:
                reward_cost_logger(reward_cost.item())
            if entropy_cost_logger is not None:
                entropy_cost_logger(entropy_cost.item())
            for callback in network_callbacks:
                callback(dags, log_probs)

    return batch_scores

def do_generative_graph_modeling(dag_model, graph_dl, 
                                 optimizer, epochs, nll_callback=None, callbacks_with_logsteps=[]):
    """ Train a DAG model on the given dataset of graphs. 
        `dag_model`: a model defining a likelihood function over graphs. Should implement
        a log_prob() method.
        `graph_dl`: a dataloader which yields graphs.

    """

    nll_loss = []

    for ep in range(epochs):
        for batchindex, graph_batch in enumerate(graph_dl):
            print(batchindex)
            loss = - dag_model.log_probs_from_batchdag(graph_batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_rec = loss.detach().item()
        nll_loss.append(loss_rec)
        if nll_callback is not None:
            nll_callback(loss_rec)
        
        for logstep, callback in callbacks_with_logsteps:
            if ep % logstep == 0:
                callback(dag_model)
        
        print(f"Finished epoch {ep}")
    print("Training complete")
    return nll_loss
