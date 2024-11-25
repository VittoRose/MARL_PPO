# MAPPO with parameter sharing

import gymnasium as gym
import torch
from grid_env.coverage import encode_action, decode_reward

from MAPPO.buffer import Buffer
from MPPO.ActorCritic import Agent
from MAPPO.parameters import *
from MAPPO.utils.run_info import InfoPlot
from MAPPO.utils.util_function import make_env
import MAPPO.algo as PS

# Run name for logger, use None if no logger is needed
name = None

# Tensorboard Summary writer
gym_id = "GridCoverage-v0"
logger = InfoPlot(gym_id, name, "cpu", folder="logs/")

# Environments for training and
envs = gym.vector.SyncVectorEnv([make_env(gym_id, n_agent=2) for _ in range(N_ENV)])
test_env = gym.make(gym_id, n_agent=2, map_id=1)

# Environment spaces
obs_shape = 33
action_shape = 5

# Use a single agent to represent two agent with parameter sharing
agent = Agent(obs_shape, action_shape)
optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

# Get the first observation
next_obs, _ = envs.reset(seed=SEED)
next_obs = torch.tensor(next_obs)
next_done = torch.zeros(N_ENV)

for epoch in range(0, MAX_EPOCH):
    
    # Progress bar
    logger.show_progress(epoch)

    # Test agent in a separate environment
    IPPO.test_network(epoch, agent, agent, test_env, logger)
    
    # Collect data from the environment
    for step in range(0, N_STEP):

        buffer0.update(next_obs[:,0,:], next_done, step)
        buffer1.update(next_obs[:,1,:], next_done, step)

        # Get action and value from current state
        with torch.no_grad():
            action0, logprob0, _, value0 = agent0.get_action_and_value(next_obs[:,0,:])
            action1, logprob1, _, value1 = agent1.get_action_and_value(next_obs[:,0,:])
            
            action = encode_action(action0.cpu(), action1.cpu())

        # Execute action in environment
        next_obs, code_reward, truncated, terminated, _ = envs.step(action)
                    
        reward0, reward1 = decode_reward(code_reward)
        
        done = terminated | truncated

        buffer0.store(value0, action0, logprob0, reward0, step)
        buffer1.store(value1, action1, logprob1, reward1, step)
        
        next_obs, next_done = torch.tensor(next_obs), torch.tensor(done)

    advantages0, returns0 = IPPO.get_advantages(agent0, buffer0, next_obs[:, 0, :], next_done)
    advantages1, returns1 = IPPO.get_advantages(agent0, buffer0, next_obs[:, 1, :], next_done)

    # flatten the batch
    b_advantages0 = advantages0.reshape(-1)
    b_advantages1 = advantages1.reshape(-1)
    b_returns0 = returns0.reshape(-1)
    b_returns1 = returns1.reshape(-1)

    IPPO.update_network(agent0, optimizer0, buffer0, b_advantages0, b_returns0, logger, 0)
    IPPO.update_network(agent1, optimizer1, buffer1, b_advantages1, b_returns1, logger, 1)

test_env.close()                
envs.close()
logger.close()
