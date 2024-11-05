import numpy as np
import gymnasium as gym
from gymnasium import spaces

#-------------------------------#
#           Reward
#-------------------------------#

# Agent move into a wall
CONTACT = -5

# Agent move in a empty tile
TILE_NOT_COVERED = 10

# Agent moved in a previous covered tile
TILE_COVERED = -1

# Agent move out map
OUT = CONTACT

# Agent bump into other agent
COLLISION = -10

# Agents covered all tiles
ALL_COVERED = 100

#-------------------------------#
#           Env param
#-------------------------------#

N_AGENT = 2
DTYPE = np.int8

OBSTACLE = -1
FREE = 0
VISITED = 1

class GridCoverage(gym.Env):
    """
    Class that contain the rules for the multi agent enviroment
        The map is represented by a grid with a value for each tile:
            0   -> Tile not visited
            -1  -> Tile with obstacle
            1   -> Visited tile
            2   -> Agent in that tile
    """
    
    def __init__(self, map_id: int):

        super(GridCoverage, self).__init__()

        self.map_id = map_id
        
        if self.map_id == 1:
            self.h = 5
            self.w = 5
        else:
            raise ValueError("Map not available")

        self.action_space = spaces.MultiDiscrete([5,5])
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(N_AGENT, 2*N_AGENT+4+self.h*self.w),
            dtype=np.float32
        )

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

        # Position on the grid for both agents
        self.agent_xy = np.zeros((N_AGENT, 2), dtype=DTYPE)
        self.grid = np.zeros((self.h, self.w), dtype=DTYPE)

        # Place obstacle
        if self.map_id == 1:
            self.grid[1,0] = OBSTACLE
            self.grid[1,1] = OBSTACLE
            self.grid[2,3] = OBSTACLE
            self.grid[3,3] = OBSTACLE
            self.grid[3,2] = OBSTACLE
            
            # Min and max value from the grid
            self.min_grid = self.grid.sum()
            self.max_grid = self.w*self.h - self.min_grid
        else:
            raise ValueError("Map not available")
        
        # Agents initial position
        while True:
            pos1 = (np.random.randint(0, self.w), np.random.randint(0, self.h))
            if self.grid[pos1] == FREE:
                # Mark the tile as visited by the agent 0 (value == 1)
                self.grid[pos1] = VISITED
                break

        while True:
            pos2 = (np.random.randint(0, self.w), np.random.randint(0, self.h))
            if self.grid[pos2] == FREE:
                # Mark the tile as visited by the agent 1 (value == 2)
                self.grid[pos2] = VISITED + 1
                break        

        # Store agent position
        self.agent_xy[0] = pos1
        self.agent_xy[1] = pos2
        
        info = {}
        obs = self._get_obs()
        
        return obs, info


    def step(self, action: list[int]) -> tuple[tuple[np.array], tuple[int], bool, bool, None]:
        """
        Action: 0 -> Hold
                1 -> Up
                2 -> Down
                3 -> Left
                4 -> Right
        """
        
        reward = [0, 0]

        terminated, truncated = False, False

        for agent in range(N_AGENT):
            
            act = action[agent]

            skip, key = self.act2key(act)
            
            # Execute action and get the reward
            if not skip:
                if self.check_obstacle(self.agent_xy[agent], self.offsets[key]):
                    reward[agent] += CONTACT
                
                elif self.check_out(self.agent_xy[agent], self.offsets[key]):
                    reward[agent] += OUT
                
                elif self.check_collision(self.agent_xy[agent], self.offsets[key], agent):
                    reward[agent] += COLLISION
                
                elif self.check_visited(self.agent_xy[agent], self.offsets[key]):
                    reward[agent] += TILE_COVERED
                    # Move agent
                    self.agent_xy[agent] += self.offsets[key]
                    # Set new position as visited from this agent
                    to_update = np.asarray(self.agent_xy[agent])
                    self.grid[to_update[0], to_update[1]] = VISITED + agent
                
                else:
                    reward[agent] += TILE_NOT_COVERED
                    # Move agent
                    self.agent_xy[agent] += self.offsets[key]
                    
                    # Set new position as visited
                    to_update = np.asarray(self.agent_xy[agent])
                    self.grid[to_update[0], to_update[1]] = VISITED + agent

        # Check if all tiles are covered
        if self.get_coverage().sum() >= self.max_grid:
            reward[0] += ALL_COVERED
            reward[1] += ALL_COVERED
            terminated = True

        # Prepare out variable
        obs = self._get_obs()
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.array:
        """
        Return the observation for reset() and step()
            Single state configuration:
                Self position (normalized)                  [x,y]                                       type = float
                Position of other agent (normalized)        [x1, y1]                                    type = float
                Presence of obstacle around the agent:      [Up, Down, Left, Right]                     type = bool
                Tiles covered:                              [Tile[0,0], Tile[0,1], .... Tile[h+1,w+1]]  type = bool
        """
        # Preallocation
        obs = np.asarray(np.zeros((N_AGENT, 2+2+4+self.w*self.h), dtype=np.float32))

        for i in range(N_AGENT):           
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

        up = self.check_obstacle(pos, self.offsets["up"])
        down = self.check_obstacle(pos, self.offsets["down"])
        left = self.check_obstacle(pos, self.offsets["right"])
        right = self.check_obstacle(pos, self.offsets["left"])

        return np.asarray((up, down, left, right))

    def check_obstacle(self, pos, offset) -> bool:
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
    
    def check_out(self, pos, offset) -> bool:
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
        
    def check_collision(self, pos, offset, agent) -> bool:
        """
        Check if the agent is tring to move to a tile already occupied by the other agent
        """
        p = np.asarray(pos)
        o = np.asarray(offset)
        to_check = p+o

        for other in range(N_AGENT):
            if agent != other:
                # print(f"Roomba che attacca: {agent}, rumba fermo: {other}")
                collision = 1 if (to_check[0] == self.agent_xy[other][0]) and (to_check[1] == self.agent_xy[other][1]) else 0

        return collision
        
    def check_visited(self, pos, offset) -> bool:
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