"""
This file show the gui and let you control the agents manually using keyboard
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
    actions = []  
    
    while len(actions) < 2: 
        for event in pg.event.get():
            if event.type == pg.QUIT: 
                pg.quit()
                
            if event.type == pg.KEYDOWN: 
                if event.key in allowed_action: 
                    azione = allowed_action[event.key]
                    actions.append(azione)
                    if len(actions) == 2: 
                        break

    return actions[0], actions[1] 

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
                            
        a1, a2 = chose_action()

        action = encode_action(torch.tensor(a1), torch.tensor(a2))
        state, reward, term, trunc, _ =  env.step(action)
        
        if term or trunc: 
            print(term, trunc)
            
        time.sleep(.1)
        
        screen.update(env, [a1,a2])