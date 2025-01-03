"""
This file is used to test trained network with a GUI

Change manually the path to the agent parameters

Choose the algorithm used to train the network with the ALGO param
"""

ALGO = "MAPPO"
ALGO = "PPO"
path = "Saved_agents/Debug_1_PPO.pth"

# ALGO = "IPPO"
path0 = "Saved_agents/Agent_0_39.pth"
path1 = "Saved_agents/Agent_1_25.pth"

import pygame as pg
import torch
import numpy as np
from grid_env.coverage import GridCoverage, encode_action
from grid_env.gui import GUI

obs_shape = 33
action_shape = 5

if ALGO == "IPPO":
    from IPPO.ActorCritic import Agent
    
    agent0 = Agent(obs_shape, action_shape)
    agent1 = Agent(obs_shape, action_shape)

    agent0.eval()
    agent1.eval()

    agent0.load(path0)
    agent1.load(path1)
    
elif ALGO == "MAPPO":
    from MAPPO.ActorCritic import Networks
    
    agent0 = Networks(obs_shape, action_shape)
    agent1 = Networks(obs_shape, action_shape)

    agent0.load(path)
    agent1.load(path)
    
elif ALGO == "PPO":
    from PPO.ActorCritic import Agent
    
    agent = Agent(obs_shape-2, action_shape)
    agent.load(path)
    
else:
    raise NameError(f"Algorithm {ALGO} not supported")

if __name__ == "__main__":
    import time

    env = GridCoverage(1,1) if ALGO == "PPO" else GridCoverage(2,1)
    state, _ = env.reset()
    screen = GUI(env)
    screen.update(env, (0,0))
    step = 0
    run = True
    time.sleep(1)
    
    while run:
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
        
        
        if ALGO == "PPO":       # Get action for one agent
            action = agent.get_action_test(torch.as_tensor(state))
            a1 = action
            a2 = None
            
        else:                   # Get action for two agents and encode it
            a1 = agent0.get_action_test(torch.as_tensor(state[0]))
            a2 = agent1.get_action_test(torch.as_tensor(state[1]))

            action = encode_action(a1, a2)
            
        state, reward, term, trunc, _ =  env.step(action)
        step += 1
        
        screen.update(env, (a1,a2), step)
        
        if term or trunc:
            state, _ = env.reset()
            step = 0
            time.sleep(2)
            screen.update(env, (a1,a2), step)
                               
        time.sleep(1.5)
        