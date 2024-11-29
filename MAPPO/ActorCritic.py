import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
import os
import numpy as np

# Number of hidden layer, neurons and activation funcion are defined in parameters.py
from .parameters import N_LAYER, N_NEURONS, ACT_FN, N_ENV

def init(module, weight_init, bias_init, gain=1):
    """ Init for weight and bias """
     
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

class MLP(nn.Module):
    """
        Base class with the first layer and all the hidden layer for Actor and Critic network
    """
    def __init__(self, input_dim):
        
        super(MLP, self).__init__()
        self.n_layer = N_LAYER
        
        if isinstance(ACT_FN, nn.Tanh):
            gain = nn.init.calculate_gain('tanh')
        else: 
            raise NotImplementedError("Manually change gain evaluation for activation function different from Tanh")
        
        def layer_init(layer):
            return init(layer, nn.init.orthogonal_, lambda x: nn.init.constant_(x,0), gain)
        
        # First Fully connected layer 
        self.fc1 = nn.Sequential(layer_init(nn.Linear(input_dim, N_NEURONS)), 
                                 ACT_FN,
                                 nn.LayerNorm(N_NEURONS))
        
        # Hidden layer
        self.fc2 = nn.ModuleList([nn.Sequential(
            layer_init(nn.Linear(N_NEURONS, N_NEURONS)),
            ACT_FN,
            nn.LayerNorm(N_NEURONS))
                                  for i in range(self.n_layer)])
        
    def forward(self, state):
        x = self.fc1(state)
        for i in range(self.n_layer):
            x = self.fc2[i](x)
        return x
            
class Actor(nn.Module):
    """
    Actor Network for MAPPO algorithm.
    """
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        
        # First + hydden layer
        self.base = MLP(input_dim)
        
        # Last layer
        self.last = nn.Sequential(nn.Linear(N_NEURONS, output_dim))
        
    def forward(self, state):
        """
        Output the policy for the given state
        """
        x = self.base(state)
        out = self.last(x)
        return out
    
    def get_action(self, x, action=None):
        """
        Samples or evaluates an action from the policy distribution.

        Parameters:
        x (Tensor): Input state.
        action (Tensor, optional): Action to evaluate. If None, an action is sampled from the policy.

        Returns:
        action (Tensor): The selected action.
        log_prob (Tensor): Log probability of the action.
        entropy (Tensor): Entropy of the policy distribution.
        """

        x = self.base(x)
        logits = self.last(x)
        
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy
        
class Critic(nn.Module):
    """
    Critic Network for MAPPO algorithm
    """
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        
        # First + hydden layer
        self.base = MLP(input_dim)
        
        # Last layer
        self.last = nn.Sequential(nn.Linear(N_NEURONS, 1))
        
    def forward(self, obs):
        """
        Compute the value of the given observation
        """
        x = self.base(obs)
        out = self.last(x)
        return out
    
class Networks():
    """
    Wrappers for Actor and Critic networks
    """
    
    def __init__(self, state_dim, action_dim, lr_list = None, critic_state_dim = None, device=torch.device("cpu")):
        
        self.state_dim = state_dim
        
        self.actor = Actor(self.state_dim, action_dim)
        
        if lr_list is not None:
            self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_list[0])
        
        if critic_state_dim is not None:
            self.critic_state_dim = critic_state_dim
            self.critic = Critic(self.critic_state_dim)
            self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_list[1])
        
        
    def get_action_value(self, states: torch.tensor, critic_state: torch.tensor) -> tuple[torch.tensor]:
        """
        Evaluate the current state with actor and critic network, used during policy rollout
        :param states: batch of states for actor network [N_AGENT, N_ENV, OBS]
        :param critic_states: batch of states for critic network [N_ENV, CRITIC_OBS]
        
        :return actions: action tensor with shape [N_ENV, N_AGENT]
        :return logprobs: tensor with logprob of each action in actions, shape [N_ENV, N_AGENT]
        :return values: tensor with values for each critic state, shape [N_ENV]
        """
        
        actions = torch.zeros(2, N_ENV)
        logprobs = torch.zeros(2, N_ENV)
        
        # Select the action for each agent with the same actor
        for i, state in enumerate(states):
            action, logprob, _ = self.actor.get_action(state)
            actions[i] = action
            logprobs[i] = logprob 
        
        values = self.critic(critic_state)
        
        return torch.t(actions), torch.t(logprobs), values.squeeze()
    
    def get_value(self, critic_state):
        """ 
        Get the value prediction from the current state 
        
        :param critic_state: Centralized critic state
        
        :return value_prediction: prediction of current state value, shape [N_ENV]
        """
        return torch.t(self.critic(critic_state)).squeeze()
    
    def evaluate_action(self, state, actions):
        """ 
        Evaluate action and value for the given state, no action output, used in update
        
        :param state: state to evaluate again with the new network [MINIBATCH_SIZE, OBS]
        
        :return probs: new logprob [MINIBATCH_SIZE]
        :return entr: new entropy [MINIBATCH_SIZE]
        """
    
        logits = self.actor(state)
        probs = Categorical(logits=logits)
        
        return probs.log_prob(actions), probs.entropy()
    
    def get_action_test(self, state):
        
        logits = self.actor(state)
        
        return torch.argmax(logits)
    
    def save_actor(self, name: str) -> None:
        """
        Save parameters for actor network in 'Saved_agents' folder, if folder doesn't exist, create one 
        """
        if name is not None:
            if not os.path.exists("Saved_agents/"):
                os.mkdir("Saved_agents/")
            
            agent_path = "Saved_agents/" + name + "shared" + ".pth"
            
            # Add a random number at the end of the name to avoid ovewrite old models
            if os.path.exists(agent_path):
                print("Save name changed, new name:")
                while os.path.exists(agent_path):
                    rng = np.random.randint(50)
                    agent_path = "Saved_agents/" + name + "shared" + "_" + str(rng) + ".pth"
                print(agent_path)
                
            torch.save(self.actor.state_dict(), agent_path)
            
    def load(self, path: str):
        """
        Load actor network
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint)
        