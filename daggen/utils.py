import graphviz

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
