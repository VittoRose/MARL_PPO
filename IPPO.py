import gymnasium as gym
import torch
import grid_env

from IPPO.buffer import Buffer
from IPPO.ActorCritic import Agent
from IPPO.parameters import *
from utils.run_info import InfoPlot
from utils.util_function import make_env
import IPPO.algo as IPPO

name = "prova_01"
gym_id = "GridCoverage-v0"

# Tensorboard Summary writer
logger = InfoPlot(gym_id, name, "cpu")

# Vector environment object
envs = gym.vector.SyncVectorEnv([make_env(gym_id, n_agent=2) for _ in range(n_env)])
test_env = gym.make(gym_id, n_agent=1, map_id=1)

# Enviroment spaces
obs_shape = 33
action_shape = 5

# Agents network and compagnia bella
agent0 = Agent(obs_shape, action_shape)
agent1 = Agent(obs_shape, action_shape)

optimizer0 = torch.optim.Adam(agent0.parameters(), lr=LR, eps=1e-5)
optimizer1 = torch.optim.Adam(agent1.parameters(), lr=LR, eps=1e-5)

buffer0 = Buffer(obs_shape, action_shape)
buffer1 = Buffer(obs_shape, action_shape)

# Get the first observation
next_obs, _ = envs.reset(seed=SEED)
next_obs = torch.tensor(next_obs)
next_done = torch.zeros(n_env)

for epoch in range(0, MAX_EPOCH):
    
    # Progress bar
    logger.show_progress(epoch)

    # Test agent in a separate environment
    IPPO.test_network(epoch, agent, test_env, logger)
    
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