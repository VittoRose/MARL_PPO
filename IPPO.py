import gymnasium as gym
import torch
import grid_env
from grid_env.vector_env import VectorEnv
from grid_env.coverage import GridCoverage , encode_action

from IPPO.buffer import Buffer
from IPPO.ActorCritic import Agent
from IPPO.parameters import *
import grid_env.vector_env
from utils.run_info import InfoPlot
from utils.util_function import make_env
import IPPO.algo as IPPO

name = None
gym_id = "GridCoverage-v0"

# Select device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device_name = "cuda" if torch.cuda.is_available() else "cpu"

# Tensorboard Summary writer
logger = InfoPlot(gym_id, name, "cpu", "IPPO")

# Vector environment object
envs = gym.vector.SyncVectorEnv([make_env(gym_id, n_agent=2) for _ in range(n_env)])
#envs = VectorEnv([GridCoverage(n_agent=2,map_id=1) for _ in range(n_env)])

# envs = gym.make(gym_id, n_agent=2, map_id=1)
test_env = gym.make(gym_id, n_agent=2, map_id=1)

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
next_obs, _ = envs.reset()

next_obs = torch.tensor(next_obs)
next_done = torch.zeros(n_env)

for epoch in range(0, MAX_EPOCH):
    
    # Progress bar
    logger.show_progress(epoch)

    # Test agent in a separate environment
    IPPO.test_network(epoch, agent0, agent1, test_env, logger)
    
    # Collect data from the environment
    for step in range(0, n_step):
        
        buffer0.update(next_obs[:,0,:], next_done, step)
        buffer1.update(next_obs[:,1,:], next_done, step)

        # Get action and value from current state
        with torch.no_grad():
            action0, logprob0, _, value0 = agent0.get_action_and_value(next_obs[:,0,:])
            action1, logprob1, _, value1 = agent1.get_action_and_value(next_obs[:,0,:])
            
            action = encode_action(action0.cpu(), action1.cpu())

        # Execute action in environment
        # TODO: solve step 
        next_obs, _, truncated, terminated, reward = envs.step(action)
        done = terminated | truncated

        buffer0.store(value0, action0, logprob0, reward[0], step)
        buffer1.store(value1, action1, logprob1, reward[1], step)
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