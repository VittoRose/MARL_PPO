import torch.nn as nn
from torch.distributions.categorical import Categorical

# Number of hidden layer, neurons and activation funcion are defined in parameters.py
from parameters import N_LAYER, N_NEURONS, ACT_FN

def init(module, weight_init, bias_init, gain=1):
    """ Init for weight an bias """
     
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
            
        return action, action.log_prob(action), probs.entropy
        
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