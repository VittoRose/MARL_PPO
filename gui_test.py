import pygame as pg
import torch
import numpy as np
from grid_env.coverage import GridCoverage, decode_reward, encode_action
from grid_env.gui import GUI

# plotjugger
# Run a demo of the gui where the agents move random
if __name__ == "__main__":
    import time

    env = GridCoverage(2,1)
    env.reset()
    screen = GUI(env)
    run = True

    while run:
        time.sleep(2)
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
                
        a1 = np.random.randint(5)
        a2 = np.random.randint(5)

        action = encode_action(torch.tensor(a1), torch.tensor(a2))
        _, reward, _, _, _ =  env.step(action)
        
        screen.update(env, [a1,a2])