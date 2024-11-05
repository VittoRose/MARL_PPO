import gymnasium as gym
import numpy as np
import torch

from PPO.buffer import Buffer
from PPO.ActorCritic import Agent_gym
from PPO.parameters import *
from utils.run_info import InfoPlot
from utils.util_function import make_env
import PPO

name = "save_space1"
gym_id = "CartPole-v1"

# Tensorboard Summary writer
logger = InfoPlot(gym_id, name, "cpu")

# Vector environment object
envs = gym.vector.SyncVectorEnv([make_env(gym_id, 0) for _ in range(n_env)])
test_env = gym.make("CartPole-v1")

# RL agent and optimizer
agent = Agent_gym(envs)
optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

# Create a buffer to store transition data
buffer = Buffer(envs)

# Get the first observation
next_obs, _ = envs.reset(seed=SEED)
next_obs = torch.tensor(next_obs)
next_done = torch.zeros(n_env)

for epoch in range(0, MAX_EPOCH):
    
    # Progress bar
    logger.show_progress(epoch)

    # Test agent in a separate environment
    PPO.test_network(epoch, agent, test_env, logger)
    
    # Collect data from the environment
    for step in range(0, n_step):

        buffer.update(next_obs, next_done, step)

        # Get action and value from current state
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)

        # Execute action in environment
        next_obs, reward, truncated, terminated, _ = envs.step(action.numpy())
        done = terminated | truncated

        buffer.store(value, action, logprob, reward, step)
        next_obs, next_done = torch.tensor(next_obs), torch.tensor(done)

    advantages, returns = PPO.get_advantages(agent, buffer, next_obs, next_done)

    # flatten the batch
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)

    PPO.update_network(agent, optimizer, buffer, b_advantages, b_returns, logger)

test_env.close()                
envs.close()
logger.close()

print("\nTraining over")