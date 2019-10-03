from config import model_params, env_params, training_params
import gym
import sys
sys.path.append('../..')
from daggen.models import GraphGRU
from daggen.utils import do_score_training
from daggen.training import get_return
import multiprocessing
import torch
import random
import numpy as np
import json
import os
import logging


def get_env(seed):
    env = gym.make(env_params.env_name)
    env.seed(seed)
    return env

def get_model(seed, min_intermediate):
    torch.manual_seed(seed)
    model = GraphGRU(model_params.input_dim, model_params.output_dim, 
                    model_params.hidden_size, model_params.logits_hidden_size,
                    len(model_params.activation_labels), 
                    min_intermediate_vertices=min_intermediate, 
                    max_intermediate_vertices=model_params.max_intermediate_vertices)
    model.set_activation_functions(model_params.activation_labels)
    return model

def do_model_training(seed, min_intermediate, batch_size, lr, tr_id):
    np.random.seed(seed)
    random.seed(seed)
    env = get_env(seed)
    model = get_model(seed, min_intermediate)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    score_fn = lambda dag: get_return(dag, env, env_params.max_timesteps,
                stochastic=training_params.stochastic)

    entropies = []
    sizes = []
    entropy_logger = lambda e: entropies.append(e)

    def size_logger(update_index, dags, log_probs):
        sizes.append( sum([d.size for d in dags]) / len(dags) )
    callbacks = [size_logger]
    batch_scores = do_score_training(model, score_fn, training_params.episodes, 
                    batch_size=batch_size, optimizer=optimizer, baseline=training_params.baseline, 
                    entropy_logger=entropy_logger, 
                    network_callbacks=callbacks)
    

    outputs = {'scores': batch_scores, 
                'sizes': sizes, 
                'entropies': entropies, 
                'id': tr_id, 
                'seed': seed}

    with open(os.path.join('data', f'outputs_{tr_id}.json'), 'w') as f:
        json.dump(outputs, f)
    

if __name__ == "__main__":

    ncore = int(sys.argv[1])
   
    nlr = len(training_params.lrvals)
    nseed = training_params.numseed
    nbatchsize = len(training_params.batch_sizes)
    nmin = len(model_params.min_intermediate_vertices)
    ntot = nlr * nseed * nbatchsize * nmin
    
    logging.basicConfig(level=logging.INFO)
    logging.info(f"found {ntot} parameter settings")
    logging.info(f"using {ncore} processes") 

    args = []

    for i in range(nmin):
        for j in range(nbatchsize):
            for k in range(nlr):
                for l in range(nseed):
                    tr_id = f"nmin_{i}_bs_{j}_lr_{k}_seed{l}"
                    args.append((random.randint(0, 10000), 
                                    model_params.min_intermediate_vertices[i],
                                    training_params.batch_sizes[j],
                                    training_params.lrvals[k],tr_id))
    
    logging.info("now training...")
    with multiprocessing.Pool(ncore) as p:
        p.starmap(do_model_training, args)
