""" Test how long DAG forward pass takes. """
import time
import numpy as np
import sys
import torch
from sacred import Experiment
from sacred.observers import MongoObserver
from torch.distributions import Categorical
sys.path.append('../..')
from daggen.models import DAG


expt = Experiment(name="dag_forward_timing")
expt.observers.append(MongoObserver.create())

@expt.config
def config():
    # which graph sizes to test
    num_intermediate_vertices = list(range(0, 50))
    #graph input dimension
    input_dim = 1
    #graph output dimension
    output_dim = 1
    #activation functions to use
    activation_functions = [lambda x : x, lambda x : -x, lambda x : x.relu(), lambda x : torch.ones_like(x)]
    #number of possible activation functions
    num_activations = len(activation_functions)
    #number of forward passes to use per graph timing
    num_forward_pass = 10
    # number of timings per size
    num_samples = 10
    #batch size of DAG inputs
    batch_size = 10

@expt.capture
def build_fully_connected_dag(input_dim, output_dim, num_intermediate, 
                    activation_functions, num_activations):
    """ Builds a DAG of the requested size where each node is connected to all nodes listed before it. """
    num_receiving = output_dim + num_intermediate
    num_emitting = input_dim + num_intermediate
    connections = torch.ones(num_receiving, num_emitting, dtype=torch.uint8)
    for i in range(num_receiving):
        connections[i, input_dim+i:] = 0
    act_distribution = Categorical(logits=torch.ones(num_activations))
    activations = act_distribution.sample((num_receiving,))

    return DAG(input_dim, output_dim, num_intermediate, connections, activations, 
            activation_functions = activation_functions)

@expt.capture
def time_forward_pass(dag, num_forward_pass, batch_size):
    inputs = torch.randn(batch_size, dag.input_dim)
    t0 = time.time()
    for __ in range(num_forward_pass):
        y = dag.forward(inputs)
    t1 = time.time()
    return (t1 - t0) / num_forward_pass

@expt.automain
def do_timing_test(num_intermediate_vertices, 
                    num_samples, seed):
    """ Build a graph and time how long it takes to perform a fixed number of forward passes."""

    torch.manual_seed(seed)
    nsize = len(num_intermediate_vertices)
    times = np.empty((nsize, num_samples))
    for i in range(nsize):
        ni = num_intermediate_vertices[i]
        for j in range(num_samples):
            print(f"Timing DAG {j} with {ni} intermediate vertices")
            dag = build_fully_connected_dag(num_intermediate=ni)
            times[i, j] = time_forward_pass(dag)
    np.save("times", times)

