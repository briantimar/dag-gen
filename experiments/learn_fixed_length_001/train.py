from sacred import Experiment
from sacred.observers import MongoObserver

import numpy as np
import torch

learn_length = Experiment(name="learn_fixed_length_001")
# learn_length.observers.append(MongoObserver.create())

@learn_length.config
def model_config():

    #graph gru hidden size
    hidden_size = 10
    #graph input dimension
    input_dim = 1
    #graph output dimension
    output_dim = 1
    #hidden size for MLP that computes logits
    logits_hidden_size = hidden_size
    #number of activation functions
    num_activations = 1 

    min_intermediate_vertices = None
    max_intermediate_vertices = 8

@learn_length.config
def training_config():
    # desired number of intermediate vertices
    target_num_intermediate = 4
    # how many graphs to sample per training step
    batch_size = 5
    # how many samples to draw from model
    total_samples = 100
    #learning rate
    learning_rate = .01

@learn_length.automain
def train(hidden_size, input_dim, output_dim, logits_hidden_size, num_activations, 
            min_intermediate_vertices, max_intermediate_vertices,
        target_num_intermediate, batch_size, total_samples, learning_rate
        ):
    """ Train a graphgru to only output graphs of a given size."""
    #build the model
    from config import GraphGRU, do_score_training
    dag_model = GraphGRU(input_dim, output_dim, hidden_size, logits_hidden_size, num_activations)
    dag_model.activation_functions = [lambda x: x]
    ## define score function for a given graph
    def score(graph):
        return np.abs(graph.get_num_intermediate() - target_num_intermediate, dtype=float)
    
    optimizer = torch.optim.Adam(dag_model.parameters(), lr=learning_rate)
    score_logger = lambda s: print(f"Batch score {s}")
    do_score_training(dag_model, score, total_samples, batch_size, optimizer, 
                            min_intermediate_vertices=min_intermediate_vertices, 
                            max_intermediate_vertices=max_intermediate_vertices,
                            score_logger=score_logger)