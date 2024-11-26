import pygame as pg
import torch
import numpy as np
from grid_env.coverage import GridCoverage, decode_reward, encode_action
from grid_env.gui import GUI
from IPPO.ActorCritic import Agent

path0 = "Saved_agents/Agent_0.pth"
path1 = "Saved_agents/Agent_1.pth"

obs_shape = 33
action_shape = 5


agent0 = Agent(obs_shape, action_shape)
agent1 = Agent(obs_shape, action_shape)

agent0.eval()
agent1.eval()

agent0.load(path0)
agent1.load(path1)

if __name__ == "__main__":
    import time

    env = GridCoverage(2,1)
    state, _ = env.reset()
    screen = GUI(env)
    run = True

    while run:
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
                
        a1 = agent0.get_action_test(torch.as_tensor(state[0]))
        a2 = agent1.get_action_test(torch.as_tensor(state[1]))

        action = encode_action(a1, a2)
        state, reward, term, trunc, _ =  env.step(action)
        
        if term or trunc: 
            print(term, trunc)
            
        time.sleep(2)
        
        screen.update(env, [a1,a2])