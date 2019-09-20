""" Learn distribution which only outputs FC graphs."""

from sacred import Experiment
from sacred.observers import MongoObserver

import numpy as np
import torch

expt = Experiment(name="learn_graph_distribution_002")
expt.observers.append(MongoObserver.create())

@expt.config
def model_config():

    #graph gru hidden size
    hidden_size = 128
    #graph input dimension
    input_dim = 2
    #graph output dimension
    output_dim = 2
    #hidden size for MLP that computes logits
    logits_hidden_size = hidden_size
    #number of activation functions
    num_activations = 1

    min_intermediate_vertices = None
    max_intermediate_vertices = None

    #intermediate size of the target graph
    target_num_intermediate = 10
    

@expt.config
def training_config():
    
    #learning rate
    learning_rate = .01
    #epochs of NLL training
    epochs = 100
    #at logging steps, how many networks to sample
    num_sample = 5
    #how frequently to sample network statistics
    network_logstep = 1


@expt.automain
def train(_run, 
        hidden_size, input_dim, output_dim, logits_hidden_size, num_activations, 
            min_intermediate_vertices, max_intermediate_vertices,
            target_num_intermediate, 
        learning_rate, epochs, num_sample, network_logstep
        ):
    
    from config import GraphGRU, do_generative_graph_modeling, dag_dataset_from_batchdag, collate_dags, build_fully_connected_graph
    from torch.utils.data import DataLoader
    print(f"learn_graph_distribution_002, id = {_run._id}")
    target_num_connections = (output_dim * (input_dim + target_num_intermediate) + sum( i + input_dim for i in range(target_num_intermediate)))
    print(f"Building model, num connections = {target_num_connections}...")
    #build the model
    dag_model = GraphGRU(input_dim, output_dim, hidden_size, logits_hidden_size, num_activations, 
                        max_intermediate_vertices=target_num_intermediate + 10)
    #set this to whatever
    dag_model.activation_functions = [lambda x: x]
    dag_model.activation_labels = ['id']

    
    optimizer = torch.optim.Adam(dag_model.parameters(), lr=learning_rate)
    
    print("Now building graph dataloader...")
    target = build_fully_connected_graph(input_dim, output_dim, target_num_intermediate)
    ds = dag_dataset_from_batchdag(target)
    dataloader = DataLoader(ds, collate_fn=collate_dags)
    assert len(dataloader) == 1

    def nll_callback(s):
        expt.log_scalar("nll", s)
    
    def log_network_stats(dag_model):
        dags, __ = dag_model.sample_networks_with_log_probs(num_sample)
        size =  np.mean([dag.size for dag in dags])
        num_connections = np.mean([dag.connections.sum().item() for dag in dags])
        expt.log_scalar("size", size)
        expt.log_scalar("num_connections", num_connections)
    
    callbacks_with_logsteps = [(network_logstep, log_network_stats)]

    print("Now starting training...")
    do_generative_graph_modeling(dag_model, dataloader, optimizer, epochs, 
                                    nll_callback=nll_callback, callbacks_with_logsteps=callbacks_with_logsteps)
    print("Training complete!")

    #after training has finished, save some of the trained graphs
    dags, __ = dag_model.sample_networks_with_log_probs(5)
    id = _run._id
    for i in range(len(dags)):
        gv = dags[i].to_graphviz()
        gv.render('plots/learned_graph_run_%d_%d'%(id,i), format='png')

