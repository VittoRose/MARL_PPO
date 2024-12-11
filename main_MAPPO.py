"""
Run this file to start a training with MAPPO on GridEnv

Modify training parameters in MAPPO.parameters

Set name = "experiment_name" to save logs and training parameters, use None to skip logs

"""
import gymnasium as gym
import torch
from grid_env.coverage import encode_action, decode_reward, get_critic_state

from MAPPO.buffer import Buffer
from MAPPO.ActorCritic import Networks
from MAPPO.parameters import *
from MAPPO.utils.run_info import InfoPlot
from MAPPO.utils.util_function import make_env
import MAPPO.algo as MAPPO

MAP_ID = 1

# Run name for logger, use None if no logger is needed
name = None

# Tensorboard Summary writer
gym_id = "GridCoverage-v0"
logger = InfoPlot(gym_id, name, "cpu", folder="logs/")

# Environments for training and
envs = gym.vector.SyncVectorEnv([make_env(gym_id, n_agent=2, map_id=MAP_ID) for _ in range(N_ENV)])
test_env = gym.make(gym_id, n_agent=2, map_id=MAP_ID)

# Environment spaces
if MAP_ID == 2:
    obs_shape = 108
elif MAP_ID == 1:
    obs_shape = 33

critic_shape = obs_shape + 4
action_shape = 5

# Use a single agent to represent two agent with parameter sharing
actor_critic = Networks(obs_shape, action_shape,  critic_shape, (LR,LR))
buffer = Buffer(obs_shape, critic_shape, action_shape, 2)

# Get the first observation
next_obs, _ = envs.reset(seed=SEED)
next_obs = torch.tensor(next_obs)
next_done = torch.zeros(N_ENV)

for epoch in range(0, MAX_EPOCH):

    # Progress bar
    logger.show_progress(epoch)

    # Test agent in a separate environment
    MAPPO.test_network(epoch, actor_critic, test_env, logger)
    
    # Collect data from the environment
    for step in range(0, N_STEP):
        
        buffer.update(next_obs, next_done, step)

        # Get action and value from current state
        with torch.no_grad():
            state_list = [next_obs[:,i,:] for i in range(2)]
            states = torch.stack(state_list)
            critic_state = get_critic_state(next_obs, N_ENV, MAP_ID)
            actions, logprob, value = actor_critic.get_action_value(states, critic_state)
            
            action = encode_action(actions[:,0].cpu(), actions[:,1].cpu())

        # Execute action in environment
        next_obs, code_reward, truncated, terminated, _ = envs.step(action)
        
        reward0, reward1 = decode_reward(code_reward)
        done = terminated | truncated

        buffer.store(critic_state, 
                     value, 
                     actions, 
                     logprob, 
                     (reward0, reward1), 
                     step)
        
        next_obs, next_done = torch.tensor(next_obs), torch.tensor(done)
        
    advantages, returns = MAPPO.get_advantages(actor_critic, buffer, next_obs, next_done, MAP_ID)
    
    MAPPO.update_network(actor_critic, buffer, advantages, returns, logger)

actor_critic.save_actor("name")

test_env.close()                
envs.close()
logger.close()
