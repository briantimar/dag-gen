"""Some training routines."""
import torch
import gym
import logging
import sys
import numpy as np


def do_episode(policy, env, max_timesteps, stop_on_done=True, render=False, 
                        **policy_kwargs):
    """ Run one episode.
        policy: a stochastic policy model. Given current states as inputs, outputs logits which define probabilities of various actions.
        env: openai gym environment.
        max_timesteps: max number of timesteps the environment is allowed to run (ie one episode)
        stop_on_done: bool, whether to stop when the environment is 'done'.
        render: bool, whether to render the environment.
        policy_kwargs: keyword args to be passed to the policy's sample_action_with_log_prob method
        Returns: states, actions, rewards , log_probs
            rewards is a list of scalars; the rest are torch.Tensors
        """
    #sample initial state from the environment
    obs = torch.tensor(env.reset(),dtype=torch.float32)
    state_trajectory = [obs]
    rewards = []
    action_trajectory = []
    log_probs = []
    
    for t in range(max_timesteps):
        if render:
            env.render()
        #sample action from policy
        action, log_prob = policy.sample_action_with_log_prob(obs, **policy_kwargs)

        action_trajectory.append(action)
        log_probs.append(log_prob)

        #update the environment 
        obs, reward, done, __ = env.step(action.numpy())
        obs = torch.tensor(obs,dtype=torch.float32)
        state_trajectory.append(obs)
        rewards.append(reward)

        if done and stop_on_done:
            break
        
    state_trajectory = torch.stack(state_trajectory)
    action_trajectory = torch.stack(action_trajectory)
    log_probs = torch.stack(log_probs)
    
    return state_trajectory, action_trajectory, rewards, log_probs

def get_rewards(policy, env, max_timesteps, stop_on_done=True, **policy_kwargs):
    """Returns rewards accumulated in a single episode"""
    __, __, rewards, __ = do_episode(policy, env, max_timesteps, stop_on_done=stop_on_done, **policy_kwargs)
    return rewards

def get_return(policy, env, max_timesteps, stop_on_done=True, **policy_kwargs):
    """Returns the scalar return obtained in a single episode"""
    return sum(get_rewards(policy, env, max_timesteps, stop_on_done=stop_on_done, **policy_kwargs))

### for asynchronous reinforce-style training.

#slave process
#note that args should be pickleable
def do_asynchronous_updates(shared_model, 
                            episode_args,
                            optimizer_constr, 
                            counter, total_updates, 
                            seed = None,
                            batch_size = 1,
                            baseline='running_average',
                            entropic_penalty = 0.,
                            
                            score_logger=None,
                            entropy_logger=None,
                            network_callbacks=[],
                            log_every=1,
                            
                            **optimizer_kwargs):
    """Perform asynchronous updates on a shared model.
        shared_model: pytorch model with shared weights
        env_name: name of the gym environment in which the DAG performance is judged
            TODO: allow this to be more general, eg a generic Task name.
        optimizer_constr: torch.optim constructor
        counter: a shared global counter. Training stops 
        """

    #build enviroment
    env_name = episode_args['env_name']
    env = gym.make(env_name)

    #set seeds
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.seed(seed)

    # create a new optimizer with local state, acting on shared parameters
    optimizer = optimizer_constr(shared_model.parameters(), **optimizer_kwargs)
    
    max_timesteps = episode_args['max_timesteps']
    stochastic = episode_args.get('stochastic', False)
    print(f"Using env {env_name} for {max_timesteps} timesteps, stochastic={stochastic}")
    print(f"Optimizer: {optimizer}")
    print(f"Seed: {seed}, batch_size = {batch_size}, total updates = {total_updates}")



    if baseline not in ("running_average", None):
        raise ValueError(f"Invalid baseline method {baseline}")
    
    def score_function(dag):
        return get_return(dag, env, max_timesteps, stochastic=stochastic)

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
    update_index = 0

    while True:
        # generate samples
        dags, log_probs = shared_model.sample_networks_with_log_probs(batch_size)
        scores = torch.tensor(list(map(score_function, dags)))

        reward_cost, entropy_cost = cost_function(scores, log_probs)
        cost = reward_cost + entropy_cost

        with counter.get_lock():
            print(f"counter: {counter.value}")
            if counter.value > total_updates:
                return 
            counter.value += 1

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
            for callback in network_callbacks:
                callback(update_index, dags, log_probs)

        update_index += 1