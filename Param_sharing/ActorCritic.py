import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import os

# Network parameters
neurons = 32
activation_fn = nn.Tanh()
hidden_layer = 2                # Only for plot, change it manually

class Agent(nn.Module):
    """
    Create a ActorCritic agent 
    :param envs: Gymnasium SyncVectorEnv, training environment
    :param init_layer: flag for initializing with weight and biases defined in layer_init
    """
    def __init__(self, obs_size, action_size, init_layer: bool = True):

        super().__init__()

        self.obs_shape = obs_size
        self.action_shape = action_size

        self.critic = make_critic(self.obs_shape, init_layer)
        self.actor = make_actor(self.obs_shape, self.action_shape, init_layer)

    def get_value(self, x):
        """ Return the value of the current state """
        return self.critic(x)

    def get_action_and_value(self, x, action=None, train: bool=True):
        """ TODO. non ho voglia adesso"""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action_test(self, x) -> int:
        """ 
        Get the action from the current state,
        return the action with the max probability from the policy
        """
        
        policy_out = self.actor(x)
        action = torch.argmax(policy_out)

        return action             
    
    def save_actor(self, name: str) -> None:
        if not os.path.exists("Saved_agents/"):
            os.mkdir("Saved_agents/")
        
        agent_path = "Saved_agents/" + name + ".pth"

        torch.save(self.actor.state_dict(), agent_path)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_critic(obs_shape, init_layer) -> torch.nn:
    """
    Function to create critic network, modify here the network structure
    """
    if init_layer:
        critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, neurons)),
            activation_fn,
            layer_init(nn.Linear(neurons, neurons)),
            activation_fn,
            layer_init(nn.Linear(neurons, 1), std=1.0),
            )
    else:
        critic = nn.Sequential(
            nn.Linear(obs_shape, neurons),
            activation_fn,
            nn.Linear(neurons, neurons),
            activation_fn,
            nn.Linear(neurons, 1),
            )
    return critic

def make_actor(obs_shape, action_shape, init_layer) -> torch.nn:

    """
    Function to create actor network, modify here the network structure
    """

    if init_layer:
        actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, neurons)),
            activation_fn,
            layer_init(nn.Linear(neurons, neurons)),
            activation_fn,
            layer_init(nn.Linear(neurons, action_shape), std=0.01)
        )
    else:
        actor = nn.Sequential(
            nn.Linear(obs_shape, neurons),
            activation_fn,
            nn.Linear(neurons, neurons),
            activation_fn,
            nn.Linear(neurons, action_shape)
        )
    return actor

class Agent_gym(Agent):
    """
    Override for agent class, used to pass only envs (vector of environment) to build a network
    It's likely that with this class you will encounter an error like "envs has no attribute..." 
    Just use class Agent and manually insert obs and action size 
    """
    def __init__(self, envs):

        obs_shape = np.array(envs.single_observation_space.shape).prod()
        action_shape = envs.single_action_space.n

        super().__init__(obs_shape, action_shape)