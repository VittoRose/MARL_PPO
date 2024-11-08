from torch.utils.tensorboard import SummaryWriter
from collections import deque
import torch
import numpy as np
import random
import os
from time import time

from PPO.parameters import *
from utils.md_report import create_md_summary, complete_md_summary

# List of supported algorithm
algo_list = ["PPO", "IPPO"]

class InfoPlot:
    """
    Class that contain the tensorboard logger and the progress bar shown during training
    """
    def __init__(self, gym_id: str,name: str, device: str, algo: str, folder: str = "logs/", rnd: bool=False) -> SummaryWriter:

        # Counter for plot
        self.loss_index = 0

        # Speed measurament variable
        self.timer = time()
        self.t0 = time()
        self.buff = deque(maxlen=100)

        # Path variable
        self.folder = folder
        self.name = name

        # Handle seed
        seed = self.set_seed(rnd)
        
        # Type of algorithm
        if algo in algo_list:
            self.algo = algo
        else:
            raise NotImplemented("Algorithm not implemented")
        
        # Loss plot
        algo_list.remove("PPO")
        if algo in algo_list: 
            self.loss_plot = [0, 0]
            self.test_index = [0, 0]
        else:
            self.loss_plot = 0
            self.test_index = 0
            
        # Add folder sintax if needed
        if folder[-1] != "/" :
            folder = folder + "/"

        print(f"Experiment name: {name}")
        print("Running on " + device)
        print("Algorithm: " + algo)
        print(f"Using seed: {seed}") 

        if name is not None:

            # Check if name is not used
            if os.path.exists(folder+name+".md") or os.path.exists(folder+name+"/"):
                raise NameError("Logger already exists, change name or folder")

            self.logger = SummaryWriter(folder + name)
            create_md_summary(gym_id, name, folder, seed, device)

        else:
            self.logger = None
        
    def add_loss(self, loss: float, tag: str = "Train/Loss") -> None:
        """
        Add loss value to tensorboard
        """ 

        if self.logger is not None:

            # Add loss only once every 50 train ep
            if self.loss_plot % 50 == 0:
                if self.algo == "PPO":
                    if type(loss) == float:
                        self.logger.add_scalar(tag, loss, self.loss_index)
                    else: 
                        self.logger.add_scalar(tag, loss.item(), self.loss_index)
                
                else:
                    raise AttributeError("Use add_loss_MARL() for Multi agent")
                
                self.loss_index += 1                    
            self.loss_plot += 1
    
    def add_loss_MARL(self, loss: float, agent: int) -> None:
        
        if self.loss_plot[agent] % 50 == 0:
            self.logger.add_scalar(f"Train/Loss {agent}", loss[agent], self.loss_index[agent])
            self.loss_index[agent] += 1
        
        self.loss_plot[agent] += 1
        
        
    def add_test(self, reward: int | list[int], length: int) -> None:
        """
        Add test reward to tensorboard
        """
        if self.logger is not None:
            if self.algo == "PPO":
                self.logger.add_scalar("Test/Reward", reward, self.test_index)
                self.logger.add_scalar("Test/Length", length, self.test_index)
            elif self.algo == "IPPO": 
                self.logger.add_scalar("Test/Reward 0", reward[0], self.test_index)
                self.logger.add_scalar("Test/Reward 1", reward[1], self.test_index)
                self.logger.add_scalar("Test/Length", length, self.test_index)
            self.test_index +=1
    
    def close(self):
        """
        Call tensorboard api, remember to call at the end of the code to avoid errors
        complete the summary by adding the total training time
        """
        if self.logger is not None:
            self.logger.flush()
            self.logger.close()
            complete_md_summary(self.folder, self.name, self.t0)

    def show_progress(self, update) -> None: 
        """
        Show progress data during training
        """
        if update != 1:
            dt = time()-self.timer
            epoch_speed = 1/dt
        else: 
            epoch_speed = 0
        
        self.timer = time()
        self.buff.append(epoch_speed)
        avg = sum(self.buff)/len(self.buff)
        remaining_time = (MAX_EPOCH-update)/(avg+1e-8)

        progress = f"\rProgress: {update/MAX_EPOCH*100:2.2f} %"
        speed = f"    Epoch/s: {epoch_speed:2.2f}"                                                  # Don't use \t
        avg_string = f"    Average speed: {avg:2.2f}"
        time_to_go = f"    Remaining time: {remaining_time/60:3.0f} min {remaining_time%60:2.0f} s"

        print(progress + speed + avg_string + time_to_go, end="")

    def set_seed(self, rnd: bool = False) -> float:
        """
        Function to set seed on all packages except for gymansium
        :param rnd: Flag for random seed, if true use time() as seed
        """
        if rnd:
            random.seed(time())
            np.random.seed(time())
            torch.manual_seed(time())
            torch.backends.cudnn.deterministic = False
            return time()
        else:
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            return SEED
        
