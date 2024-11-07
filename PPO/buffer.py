import torch
from .parameters import *

class Buffer():

    def __init__(self, observation_shape: int, action_shape: int, device: torch.device):

        self.obs_shape = (observation_shape,)
        self.action_shape = action_shape
        
        self.device = device

        # Preallocation
        self.obs = torch.zeros((n_step, n_env) + self.obs_shape).to(device)
        self.actions = torch.zeros((n_step, n_env)).to(device)
        self.logprobs = torch.zeros((n_step, n_env)).to(device)
        self.rewards = torch.zeros((n_step, n_env)).to(device)
        self.dones = torch.zeros((n_step, n_env)).to(device)
        self.values = torch.zeros((n_step, n_env)).to(device)

    def update(self, next_obs, next_done, step):
        """
        Update state and dones for the current step
        """
        self.obs[step] = next_obs.squeeze()
        self.dones[step] = next_done

    def store(self, value: torch.tensor, action: torch.tensor, logprob: torch.tensor, reward: float, step: int):
        
        with torch.no_grad():
            self.values[step] = value.flatten()
        
        self.actions[step] = action.squeeze()
        self.logprobs[step] = logprob.squeeze()
        self.rewards[step] = torch.tensor(reward).to(self.device)

    def get_batch(self):
        b_obs = self.obs.reshape((-1,) + self.obs_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_values = self.values.reshape(-1)

        return b_obs, b_logprobs, b_actions, b_values