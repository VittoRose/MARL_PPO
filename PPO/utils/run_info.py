from torch.utils.tensorboard import SummaryWriter
from collections import deque
import torch
import numpy as np
import random
import os
from time import time

from PPO.parameters import *
from .md_report import create_md_summary, complete_md_summary

# List of supported algorithm

class InfoPlot:
    """
    Class that contain the tensorboard logger and the progress bar shown during training
    """
    def __init__(self, gym_id: str,name: str, device: str, folder: str = "logs/", rnd: bool=False) -> SummaryWriter:

        # Counter for plot
        self.test_index = 0

        # Speed measurament variable
        self.timer = time()
        self.t0 = time()
        self.buff = deque(maxlen=100)

        # Handle seed
        seed = self.set_seed(rnd)
        
        # Loss plot
        self.loss_plot = 0
        self.loss_index = 0
            
        # Add folder sintax if needed
        if folder[-1] != "/" :
            folder = folder + "/"

        self.folder = folder
        
        
        if name is not None:
            summary = folder + name + ".md"
            logs = folder + name + "/"

            # Check if name is not used
            if os.path.exists(summary) or os.path.exists(logs):
                
                self.name = name
                new = 0
                
                while os.path.exists(summary) or os.path.exists(logs):
                    summary = folder + name + "_" + str(new) + ".md"
                    logs = folder + name + "_" + str(new) + "/"
                    self.name = name + "_" + str(new)
                    new += 1
                    
                print(f"Experiment name already exists, changed in: {self.name}")
                
            else: 
                print(f"Experiment name: {name}")
                
            self.logger = SummaryWriter(logs)
            create_md_summary(gym_id, name, folder, seed, device)
            # Save name for complete summary
            self.summary = summary
        else:
            self.logger = None
            

        print("Running on " + device)
        print("Algorithm: PPO")
        print(f"Using seed: {seed}") 
        
    def add_loss(self, loss: float, tag: str = "Train/Loss") -> None:
        """
        Add loss value to tensorboard
        """ 
        if self.logger is not None:

            # Add loss only once every 50 train ep
            if self.loss_plot % 50 == 0:
                if type(loss) == float:
                    self.logger.add_scalar(tag, loss, self.loss_index)
                else: 
                    self.logger.add_scalar(tag, loss.item(), self.loss_index)
                                
                self.loss_index += 1                    
            self.loss_plot += 1        
        
    def add_test(self, reward: int | list[int], length: int) -> None:
        """
        Add test reward to tensorboard
        """
        if self.logger is not None:
            self.logger.add_scalar("Test/Reward", reward, self.test_index)
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
            complete_md_summary(self.summary, self.t0)

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
        
