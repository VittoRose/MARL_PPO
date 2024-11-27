import torch
from .parameters import *

class Buffer():
    """
    Rollout buffer for PPO algorithm
    """
    
    def __init__(self, observation_shape: int, critic_observation_space: int, action_shape: int, n_agents: int):
        
        self.obs_shape = observation_shape
        self.critic_shape = critic_observation_space
        self.n_agents = n_agents
        self.action_shape = action_shape
        
        # Preallocation
        self.obs = torch.zeros((N_STEP, N_ENV, n_agents, observation_shape), dtype=torch.float32)
        
        self.actions = torch.zeros((N_STEP, N_ENV, n_agents), dtype=torch.float32)
        self.actions_log_prob = torch.zeros((N_STEP, N_ENV, n_agents), dtype=torch.float32)
        self.rewards = torch.zeros((N_STEP, N_ENV, n_agents), dtype=torch.float32)
        self.dones = torch.zeros((N_STEP, N_ENV), dtype=torch.float32)
        
        self.value_pred = torch.zeros((N_STEP, N_ENV), dtype=torch.float32)
        
    def update(self, next_obs, next_done, step):
        """
        Update state and dones for the current step
        """
        self.obs[step] = next_obs
        self.dones[step] = next_done

    def store(self, value: torch.tensor, action: torch.tensor, logprob: torch.tensor, reward: float, step: int):
        """
        Store:
            value: critic out
            action: actor choice
            logprob: log policy distr
            reward: environment reward
            step: step number in the environment
        """
        with torch.no_grad():
            self.value_pred[step] = value
        
        self.actions[step] = action
        self.actions_log_prob[step] = logprob.squeeze()
        self.rewards[step] = torch.t(torch.tensor(reward))

    def get_batch(self):
        """
        Return the batch for training
        
        :return obs: observation for each timestep, environment, agent. Shape: [N_STEP*N_ENV*N_AGENT, obs_shape]
        :return log_prob: log_prob for each timestep, environment, agent. Shape: [N_STEP*N_ENV*N_AGENT]
        :return actions: action for each timestep, environment, agent. Shape: [N_STEP*N_ENV*N_AGENT]
        :return value_pred: value for each timestep, environment. Shape: [N_STEP*N_ENV]
        """
        return self.obs, self.actions_log_prob, self.actions, self.value_pred