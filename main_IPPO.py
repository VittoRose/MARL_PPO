"""
Run this file to start a new training with Independent Proximal Policy Optimization

Modify training parameters in IPPO.parameters

Set name = "experiment_name" to save logs and training parameters, use None to skip logs

"""

import gymnasium as gym
import torch
from grid_env.coverage import encode_action, decode_reward

from IPPO.buffer import Buffer
from IPPO.ActorCritic import Agent
from IPPO.parameters import *
from IPPO.utils.run_info import InfoPlot
from IPPO.utils.util_function import make_env
import IPPO.algo as IPPO

# Run name for logger, use None if no logger is needed
name = "Debug_00"

# Tensorboard Summary writer
gym_id = "GridCoverage-v0"
logger = InfoPlot(gym_id, name, "cpu", folder="logs/")
N_AGENT = 2

# Environments for training and
envs = gym.vector.SyncVectorEnv([make_env(gym_id, n_agent=N_AGENT) for _ in range(N_ENV)])
test_env = gym.make(gym_id, n_agent=2, map_id=1)

# Enviroment spaces
obs_shape = 33
action_shape = 5

# Agents network and friend
agent, optimizer, buffer = [], [], []
for i in range(N_AGENT):
    agent.append(Agent(obs_shape, action_shape)) 
    optimizer.append(torch.optim.Adam(agent[i].parameters(), lr=LR, eps=1e-5))
    buffer.append(Buffer(obs_shape, action_shape))

# Get the first observation
next_obs, _ = envs.reset()

next_obs = torch.tensor(next_obs)
next_done = torch.zeros(N_ENV)

for epoch in range(0, MAX_EPOCH):
    
    # Progress bar
    logger.show_progress(epoch)

    # Test agent in a separate environment
    IPPO.test_network(epoch, agent[0], agent[1], test_env, logger)
    
    # Collect data from the environment
    for step in range(0, N_STEP):

        buffer0.update(next_obs[:,0,:], next_done, step)
        buffer1.update(next_obs[:,1,:], next_done, step)

        # Get action and value from current state
        with torch.no_grad():
            actions, logprobs, values = [], [], []
            for i in range(N_AGENT):
                action, logprob, _, value = agent[i].get_action_and_value(next_obs[:,i,:])
                actions.append(action)
                logprobs.append(logprob)
                values.append(value)
            action = encode_action(actions[0].cpu(), actions[1].cpu())

        # Execute action in environment
        next_obs, code_reward, truncated, terminated, _ = envs.step(action)
                    
        reward0, reward1 = decode_reward(code_reward)
        
        done = terminated | truncated

        buffer0.store(value0, action0, logprob0, reward0, step)
        buffer1.store(value1, action1, logprob1, reward1, step)
        
        next_obs, next_done = torch.tensor(next_obs), torch.tensor(done)

    advantages, returns = [], []
    for i in range(0,N_AGENT):
        advantage, returns = IPPO.get_advantages(agent[i], buffer[i], next_obs[:, i, :], next_done)

    # flatten the batch
    b_advantages0 = advantages0.reshape(-1)
    b_advantages1 = advantages1.reshape(-1)
    b_returns0 = returns0.reshape(-1)
    b_returns1 = returns1.reshape(-1)

    IPPO.update_network(agent0, optimizer0, buffer0, b_advantages0, b_returns0, logger, 0)
    IPPO.update_network(agent1, optimizer1, buffer1, b_advantages1, b_returns1, logger, 1)

# Save agents    
agent0.save_actor("Agent_0")
agent1.save_actor("Agent_1")

test_env.close()     
envs.close()
logger.close() 