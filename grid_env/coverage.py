import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from .rewards import *

DTYPE = np.int8
OBSTACLE = -1
FREE = 0
VISITED = 1

def get_critic_state(state: torch.tensor, n_env: int) -> torch.tensor:
    """
    Use the state from the two agents to get the centralized critic state
    """
    
    # Check if the state comes from one env or multiple envs
    if state.size(dim=0) == n_env:
        critic_states = torch.zeros((n_env, 37))
        for i,s in enumerate(state):
            critic_states[i] = get_critic_state(s,4)
        
        return critic_states
    
    critic_state = np.zeros(37)
    
    critic_state[0:4] = state[0, 0:4]
    critic_state[4:8] = state[0, 4:8]
    critic_state[8:12] = state[1, 4:8]
    critic_state[12:] = state[0, 8:]
    
    return torch.tensor(critic_state)

def encode_action(action_0: torch.tensor, action_1: torch.tensor) -> list[int]:
    """
    Get the action from the two agents and encode in a single value for gym API
    """    
    val = np.arange(25).reshape(5, 5) 
    
    # If actions are scalar
    if action_0.dim() == 0 and action_1.dim() == 0:
        out = val[action_0.item(), action_1.item()]
    
    # If actions are tensor
    elif action_0.dim() == 1 and action_1.dim() == 1:
        if action_0.size() != action_1.size():
            raise ValueError("action_0 e action_1 must have the same dimension")
        
        out = np.zeros(action_0.size(0))
        for i in range(action_0.size(0)):
            out[i] = val[action_0[i].int().item(), action_1[i].int().item()]
        
    return out

def encode_reward(reward_key: list[str, str], terminated: bool) -> int:
    """
    Get two reward value and encode in a single value for gym api
    """
    grid = np.arange(49).reshape(7,7)
    
    code_reward = grid[reward_code[reward_key[0]], reward_code[reward_key[1]] ]
    
    # If terminated set code as negative 
    if terminated:
        code_reward = -code_reward
    
    return code_reward

def decode_reward(code_reward: int| list[int]) -> tuple[int|list[int], int|list[int]]:
    """
    Get the reward scalar value from env.step() and get the reward for each robot
    """
    
    if isinstance(code_reward, np.ndarray):
        rewards1 = []
        rewards2 = []
        
        # Recursive call for each item
        for reward in code_reward:
            r1, r2 = decode_reward(reward)
            rewards1.append(r1)
            rewards2.append(r2)
            
        return rewards1, rewards2
    
    all_covered = False

    # If the code is negative all tiles are covered
    if code_reward < 0:
        all_covered = True
        code_reward = -code_reward
    
    index0 = code_reward // 7
    index1 = code_reward % 7
    
    reward1 = rewards[reward_decoder[index0]]
    reward2 = rewards[reward_decoder[index1]]
    
    # Add the shared reward
    if all_covered:
        reward1 += rewards["all_covered"]
        reward2 += rewards["all_covered"]
        
    return reward1 , reward2

class GridCoverage(gym.Env):
    """
    Class that contain the rules for the multi agent enviroment
        The map is represented by a grid with a value for each tile:
            0   -> Tile not visited
            -1  -> Tile with obstacle
            1 + agent_id   -> Visited tile
    """
    
    def __init__(self, n_agent: int, map_id: int):

        super(GridCoverage, self).__init__()

        self.map_id = map_id
        
        if n_agent == 1 or n_agent == 2:
            self.n_agent = n_agent
        else:
            raise ValueError("Only 1 or 2 agent")
        
        if self.map_id == 1:
            self.h = 5
            self.w = 5
        else:
            raise ValueError("Map not available")

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_agent, 2*self.n_agent+4+self.h*self.w),
            dtype=np.float32
        )

        if self.n_agent == 2:
            self.action_space = spaces.Discrete(25)
        else:
            self.action_space = spaces.Discrete(5)
            
        # Position on the grid for both agents
        self.agent_xy = np.zeros((self.n_agent, 2), dtype=DTYPE)

        # Offsets to check near tiles
        self.offsets = {
            "up": [-1, 0],
            "down": [1, 0],
            "right": [0, 1],
            "left": [0, -1]
        }
        
    def reset(self, seed = None, options = None):
        """
        Set all the tiles to not visited and create obstacle
        """
        super().reset(seed=seed)
        
        # Create grid map
        self.grid = np.zeros((self.h, self.w), dtype=DTYPE)
        
        # Place obstacle
        if self.map_id == 1:
            self.grid[1,0] = OBSTACLE
            self.grid[1,1] = OBSTACLE
            self.grid[2,3] = OBSTACLE
            self.grid[3,3] = OBSTACLE
            self.grid[3,2] = OBSTACLE
            
            # Min and max value from the grid
            n_obstacle = -self.grid.sum()
            self.max_grid = self.w*self.h - n_obstacle
            
        else:
            raise ValueError("Map not available")
        
        # Agents initial position
        for agent in range(self.n_agent):
            while True:
                pos = (np.random.randint(0, self.w), np.random.randint(0, self.h))
                if self.grid[pos] == FREE:
                    # Mark the tile as visited 
                    self.grid[pos] = VISITED + agent
                    break 

            # Store agent position
            self.agent_xy[agent] = pos   
        
        obs = self._get_obs()
        
        return obs, {}


    def step(self, action: int) -> tuple[tuple[np.array], tuple[int], bool, bool, None]:
        """
        Check documentation for reward
        Action: 0 -> Hold
                1 -> Up
                2 -> Down
                3 -> Left
                4 -> Right
        """
        
        # If action is not a list must be a single value
        # if type(action) != list:
        #     action = [action]
        
        terminated, truncated = False, False
        
        # Get the action from the table
        if self.n_agent == 2:
            actions = [action//5, action%5]
        
        rew_key = ["", ""]

        for agent in range(self.n_agent):
            
            act = actions[agent]

            skip, key = self.act2key(act)
            
            # Execute action and get the reward
            if not skip:
                if self.obstacle(self.agent_xy[agent], self.offsets[key]):
                    rew_key[agent] = "contact"
                
                elif self.out(self.agent_xy[agent], self.offsets[key]):
                    rew_key[agent] = "out"
                
                elif self.collision(self.agent_xy[agent], self.offsets[key], agent) and self.n_agent > 1:
                    rew_key[agent] = "collision"
                
                elif self.visited(self.agent_xy[agent], self.offsets[key]):
                    rew_key[agent] = "tile_covered"

                    # Move agent
                    self.agent_xy[agent] += self.offsets[key]
                    # Set new position as visited from this agent
                    to_update = np.asarray(self.agent_xy[agent])
                    self.grid[to_update[0], to_update[1]] = VISITED + agent
                
                else:
                    rew_key[agent] = "tile_not_covered"
                    # Move agent
                    self.agent_xy[agent] += self.offsets[key]
                    
                    # Set new position as visited
                    to_update = np.asarray(self.agent_xy[agent])
                    self.grid[to_update[0], to_update[1]] = VISITED + agent
            else:
                rew_key[agent] = "still"
        
        # Check if all tiles are covered
        if self.get_coverage().sum() >= self.max_grid:
            terminated = True
                
        if self.n_agent == 2:
            reward: int = encode_reward(rew_key, terminated)
        else:
            reward: int = reward_code[rew_key[0]]
            if terminated:
                reward += rewards["all_covered"]

        # Prepare out variable
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self) -> np.array:
        """
        Return the observation for reset() and step()
            Single state configuration if multi angent:
                Self position (normalized)                  [x,y]                                       type = float
                Position of other agent (normalized)        [x1, y1]                                    type = float
                Presence of obstacle around the agent:      [Up, Down, Left, Right]                     type = bool
                Tiles covered:                              [Tile[0,0], Tile[0,1], .... Tile[h+1,w+1]]  type = bool
            If single agent the state is shorter and the position of other agent is missing
        """
        # Preallocation
        obs = np.zeros((self.n_agent, 2*self.n_agent+4+self.w*self.h), dtype=np.float32)

        if self.n_agent == 2:

            for i in range(self.n_agent):           
                # Position of agent
                obs[i, 0] = self.agent_xy[i,0]/self.w
                obs[i, 1] = self.agent_xy[i,1]/self.h
                
                # Obstacle around
                obs[i, 4:8] = self.obstacle_detect(i)

            # Agent 0 get the pos of agent 1
            obs[0, 2] = self.agent_xy[1, 0]/(self.w)
            obs[0, 3] = self.agent_xy[1, 1]/(self.h)
            
            # Agent 1 get the pos of agent 0
            obs[1, 2] = self.agent_xy[0, 0]/(self.w)
            obs[1, 3] = self.agent_xy[0, 1]/(self.h)

            # Shared part of the state with the tile covered
            obs[:, 8:] = self.get_coverage()
        else:            
            obs[0,0] = self.agent_xy[0,0]/self.w
            obs[0,1] = self.agent_xy[0,1]/self.h    
            obs[0,2:6] = self.obstacle_detect(0)
            obs[0,6:] = self.get_coverage()
            
        return obs
    
    def act2key(self, action) -> tuple[bool, str]:
        """
        Map the action from the network to a key for offset dict
        Return also a bool that is true if the agent hold his position
        """

        skip = False
        key = None

        match action:
            case 0: 
                skip = True
            case 1:
                key = "up"
            case 2: 
                key = "down"
            case 3:
                key = "left"
            case 4: 
                key = "right"

        return skip, key
    
    def obstacle_detect(self, agent_index) -> np.array:
        """
        Return the obstacle part of the observation for one agent
        """
        pos = self.agent_xy[agent_index]

        up = self.obstacle(pos, self.offsets["up"])
        down = self.obstacle(pos, self.offsets["down"])
        left = self.obstacle(pos, self.offsets["right"])
        right = self.obstacle(pos, self.offsets["left"])

        return np.asarray((up, down, left, right))

    def obstacle(self, pos, offset) -> bool:
        """
        Check adiacent tile for obstacle
            Return 0 if no obstacle or no map
            Return 1 if obstacle detect
        """
        p = np.asarray(pos)
        o = np.asarray(offset)
        to_check = p+o
        try:
            if any(x<0 for x in to_check):
                return 0
            elif to_check[0] >= self.h or to_check[1] >= self.w:
                return 0
            else:
                return 1 if self.grid[to_check[0], to_check[1]] == OBSTACLE else 0
        except IndexError:
            return 0
        
    def get_coverage(self) -> np.array:
        """
        Get the last part of the observation, a list of the tiles already visited
        """
        # Reshape grid (matrix) into a vector
        temp = self.grid.flatten()

        # Sobstitute obstacle value with not visited
        temp[temp == OBSTACLE] = 0
        temp[temp >= VISITED] = VISITED
        
        return temp
    
    def out(self, pos, offset) -> bool:
        """
        Check adiacent tile for map
            Return 1 if no map
            Return 0 otherwise
        """ 
        p = np.asarray(pos)
        o = np.asarray(offset)
        to_check = p+o
        
        if any(x<0 for x in to_check) or to_check[0] >= self.h or to_check[1] >= self.w:
            return 1
        else:
            return 0
        
    def collision(self, pos, offset, agent) -> bool:
        """
        Check if the agent is tring to move to a tile already occupied by the other agent
        """
        p = np.asarray(pos)
        o = np.asarray(offset)
        to_check = p+o
        
        collision = 0

        for other in range(self.n_agent):
            if agent != other:
                collision = 1 if (to_check[0] == self.agent_xy[other][0]) and (to_check[1] == self.agent_xy[other][1]) else 0

        return collision
        
    def visited(self, pos, offset) -> bool:
        """
        Check adiacent tile if already visited
            Return 1 if tile already visited
            Return 0 otherwise
        """ 
        p = np.asarray(pos)
        o = np.asarray(offset)
        to_check = p+o

        # All error cases are already checked
        return 1 if self.grid[to_check[0], to_check[1]] >= VISITED else 0