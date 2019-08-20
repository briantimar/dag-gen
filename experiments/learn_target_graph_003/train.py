from sacred import Experiment
from sacred.observers import MongoObserver

import numpy as np
import torch
import os



expt = Experiment(name="learn_target_graph_003")
expt.observers.append(MongoObserver.create())

@expt.config
def model_config():

    #graph gru hidden size
    hidden_size = 20
    #graph input dimension
    input_dim = 3
    #graph output dimension
    output_dim = 1
    #hidden size for MLP that computes logits
    logits_hidden_size = hidden_size
    #number of activation functions
    num_activations = 3

    min_intermediate_vertices = 2
    max_intermediate_vertices =20

@expt.config
def training_config():
    # how many graphs to sample per training step
    batch_size = 1
    # how many inputs to sample at each model evaluation
    input_batch_size = 100
    # how many samples to draw from model
    total_samples = 1000
    #learning rate
    learning_rate = .01
    #number of updates between logging events
    log_every = 1

@expt.automain
def train(_run, 
        hidden_size, input_dim, output_dim, logits_hidden_size, num_activations, 
            min_intermediate_vertices, max_intermediate_vertices,
        batch_size, input_batch_size, total_samples, learning_rate, log_every
        ):
    """ Train a graphgru to only output graphs of a given size."""

    #build the model
    from config import GraphGRU, do_score_training
    dag_model = GraphGRU(input_dim, output_dim, hidden_size, logits_hidden_size, num_activations, 
                            min_intermediate_vertices=min_intermediate_vertices,
                            max_intermediate_vertices=max_intermediate_vertices)
    dag_model.activation_functions = [lambda x: x, lambda x: -x, lambda x: x.abs() , lambda x: torch.ones_like(x), lambda x: .5 * x]
    dag_model.activation_labels = ['id', 'inv', 'abs', '1', '*.5']

    #the function we want to learn
    def target_function(inputs):
        x1, x2, x3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        n1 = - (x1 + x2)
        n2 = (x2 + x3).abs()
        return (x1 + n1 + n2)

    ## define score function for a given graph
    ## THE THING YOU WANT TO MAXIMIZE!!
    ## MSE between proposed and target graph over rand-normal inputs
    def score(graph):
        inputs = torch.randn(input_batch_size, input_dim)
        target = target_function(inputs).view(input_batch_size)
        y = graph.forward(inputs).view(input_batch_size)
        return -((y - target)**2).mean().item()
    
    optimizer = torch.optim.Adam(dag_model.parameters(), lr=learning_rate)

    def score_logger(s):
        print(f"Batch score {s}")
        expt.log_scalar("score", s)
    
    def log_mean_size(dags):
        size =  np.mean([dag.size for dag in dags])
        expt.log_scalar("size", size)
    
    network_callbacks = [log_mean_size]

    do_score_training(dag_model, score, total_samples, batch_size, optimizer, 
                            network_callbacks=network_callbacks,
                            log_every=log_every,
                            score_logger=score_logger)
    
    #after training has finished, save some of the trained graphs
    dags, __ = dag_model.sample_networks_with_log_probs(10)
    id = _run._id
    pdir = os.path.join('plots', str(id))
    os.mkdir(pdir)
    scores = []
    for i in range(len(dags)):
        gv = dags[i].to_graphviz()
        gv.render(os.path.join(pdir, 'learned_graph_%d'%i), format='png')
        scores.append(score(dags[i]))
    with open(os.path.join(pdir, "scores.txt"), 'w') as f:
        for i in range(len(scores)):
            f.write(f"score {i}: {scores[i]}\n")

