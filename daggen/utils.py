import graphviz
import torch

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
            label = ''
        else:
            label = activation_labels[activations[i-input_dim]]
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

def do_score_training(dag_model, score_function, 
                        total_samples, batch_size, optimizer, 
                            min_intermediate_vertices=None,
                            max_intermediate_vertices=None, 
                            score_logger=None,
                            network_callbacks=[],
                            log_every=1,
                            baseline='running_average'):
    """
    Run score-based "policy-gradient" training on the given DAG model.
    `dag_model`: A generative model over DAGs which produces DAG objects and associated log-probability tensors
        should implement `sample_networks_with_log_probs()`
    `score_function`: a function which takes a `DAG` as input and returns a scalar (float) score.
    `total_samples`: int, total number of samples to be drawn from DAG model during training.
    `batch_size`: int, how many samples to use per update step
    `optimizer`: a torch.optim optimizer for the DAG model parameters. 

    `min_intermediate_vertices`: if not None, minimum number of vertices for each sampled graph
    `max_intermediate_vertices`: if not None, max number of vertices for each sampled graph

    `score_logger`: if not None, callback to be called on each batch score value
    `network_callbacks`: list of functions, each of which will be applied to the batch of
    dags sampled from the network at each logging step.
    `baseline`: whether and how to subtract baseline from the score functions.
        choices: ("running_average", `None`)
    `log_every`: Number of update steps between logging.
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
        return - ((scores - bl) * log_probs).sum()

    batch_scores = []

    for update_index in range(num_update):
        # generate samples
        _batch_size = get_batch_size(update_index)
        dags, log_probs = dag_model.sample_networks_with_log_probs(_batch_size)
        scores = torch.tensor(list(map(score_function, dags)))

        cost = cost_function(scores, log_probs)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        batch_score = scores.mean().item()
        batch_scores.append(batch_score)
        running_avg_score = .9 * running_avg_score + .1 * batch_score

        if update_index % log_every == 0:
            if score_logger is not None:
                score_logger(batch_score)
            for callback in network_callbacks:
                callback(dags)

    return batch_scores
