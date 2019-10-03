from collections import namedtuple

TrainingParams = namedtuple('TrainingParams', 
                            ['numseed', 'lrvals', 'episodes', 'batch_sizes', 
                            'baseline', 'stochastic'])
EnvParams = namedtuple('EnvParams', 
                        ['env_name', 'max_timesteps'])

ModelParams = namedtuple('ModelParams', 
                        ['hidden_size', 'logits_hidden_size', 'input_dim', 'output_dim', 
                        'activation_labels', 'max_intermediate_vertices', 
                        'min_intermediate_vertices'])

#number of seeds per hp setting
numseed = 10
#lr values to try
lrvals = [10**n for n in range(-4, 0)]
#how many episodes to use in training
episodes = 1000
#batch sizes to try
batch_sizes = [1, 5, 10]
#reinforce baseline
baseline = 'running_average'
#whether to use stochastic or deterministic policies
stochastic = False
#env settings
env_name = "CartPole-v0"
max_timesteps=200

hidden_size = 256
logits_hidden_size = 128
input_dim = 4
output_dim = 2
activation_labels = ['id', 'inv', 'abs', 'cos', 'bias1', 'relu', 'gauss']
max_intermediate_vertices = 30
min_intermediate_vertices = list(range(0, 4))

training_params = TrainingParams(numseed, lrvals, episodes, batch_sizes, baseline, stochastic)
env_params = EnvParams(env_name, max_timesteps)
model_params = ModelParams(hidden_size, logits_hidden_size, input_dim, output_dim, 
                            activation_labels, max_intermediate_vertices, min_intermediate_vertices)