import gymnasium as gym
import torch

import grid_env                             # Import for registration
from PPO.buffer import Buffer
from PPO.ActorCritic import Agent
from PPO.parameters import *
from utils.run_info import InfoPlot
from utils.util_function import make_env
import PPO.algo as PPO

name = "cuda_support"
gym_id = "GridCoverage-v0"

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = "cuda" if torch.cuda.is_available() else "cpu"

# Tensorboard Summary writer
logger = InfoPlot(gym_id, name, device_name)

# Vector environment object
envs = gym.vector.SyncVectorEnv([make_env(gym_id, n_agent=1) for _ in range(n_env)])
test_env = gym.make(gym_id, n_agent=1, map_id=1)

obs_shape = 31
action_shape = 5

# RL agent and optimizer
agent = Agent(obs_shape, action_shape).to(device)
optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

# Create a buffer to store transition data
buffer = Buffer(obs_shape, action_shape, device)

# Get the first observation
next_obs, _ = envs.reset(seed=SEED)
next_obs = torch.tensor(next_obs).to(device)
next_done = torch.zeros(n_env).to(device)

for epoch in range(0, MAX_EPOCH):
    
    # Progress bar
    logger.show_progress(epoch)

    # Test agent in a separate environment
    PPO.test_network(epoch, agent, test_env, logger, device)
    
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
        next_obs, next_done = torch.tensor(next_obs).to(device), torch.tensor(done).to(device)

    advantages, returns = PPO.get_advantages(agent, buffer, next_obs, next_done, device)

    # flatten the batch
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)

    PPO.update_network(agent, optimizer, buffer, b_advantages, b_returns, logger)

test_env.close()                
envs.close()
logger.close()

print("\nTraining over")