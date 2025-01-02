"""
This file show the gui and let you control the agents manually using keyboard

Instructions:

    Each timestep the code expect two input, the first one is the action for agent 0, the second one is for agent 1
    
    Controls: 
        W -> Move up
        S -> Move down
        A -> Move left
        D -> Move right 
        SPACE -> Stay still
"""

import pygame as pg
import torch
import numpy as np
from grid_env.coverage import GridCoverage, decode_reward, encode_action
from grid_env.gui import GUI


allowed_action = {
    pg.K_w: 1,
    pg.K_a: 3,
    pg.K_s: 2,
    pg.K_d: 4,
    pg.K_SPACE: 0
}

def chose_action():
    """ Wait for two different instructions """
    actions = []  
    
    while len(actions) < env.n_agent: 
        for event in pg.event.get():
            if event.type == pg.QUIT: 
                pg.quit()
                
            if event.type == pg.KEYDOWN: 
                if event.key in allowed_action: 
                    action = allowed_action[event.key]
                    actions.append(action)
                    if len(actions) == 2: 
                        break

    return actions

if __name__ == "__main__":
    import time

    env = GridCoverage(n_agent=2, map_id=2)
    state, _ = env.reset()
    screen = GUI(env)
    run = True
    step = 0
    screen.update(env, [0,0], step)

    while run:
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
                            
        a = chose_action()
        
        if env.n_agent > 1:
            action = encode_action(torch.tensor(a[0]), torch.tensor(a[1]))
        else:
            action = a[0]
            a.append(None)
        state, reward, term, trunc, _ =  env.step(action)
        
        if not term:
            step += 1
        
        if term:
            print("TERM")
        
            
        time.sleep(.1)
        
        screen.update(env, [a[0], a[1]], step)