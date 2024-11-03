import numpy as np

DTYPE = np.int8
N_AGENT = 2
OBSTACLE = -1
VISITED = 1

h,w = 5,5

agent_xy = np.asarray([[0,0],[0,0]], dtype=DTYPE)
grid = np.zeros((h,w), dtype=DTYPE)

grid[1,0] = OBSTACLE
grid[1,1] = OBSTACLE
grid[2,3] = OBSTACLE
grid[3,3] = OBSTACLE
grid[3,2] = OBSTACLE

# Agents initial position
while True:
    pos1 = (0, 0)
    if grid[pos1] != OBSTACLE:
        # Mark the tile as visited
        grid[pos1] = VISITED
        break

while True:
    pos2 = (3, 4)
    if grid[pos2] != OBSTACLE and pos2 != pos1:
        # Mark the tile as visited
        grid[pos2] = 44
        break        

# Store agent position
agent_xy[0] = pos1
agent_xy[1] = pos2

print(grid)

pos = agent_xy[0]

def check_direction(grid, pos, offset):
    """
    Check adiacent tile for obstacle
        Return 0 if no obstacle or no map
        Return 1 if obstacle detect
    """
    to_check = tuple(pos+offset)
    try:
        if any(x<0 for x in to_check):
            return 0
        else:
            return 1 if grid[to_check] < 0 else 0
    except IndexError:
        return 0

# Posizioni relative per ogni direzione
offsets = {
    "up": [-1, 0],
    "down": [1, 0],
    "right": [0, 1],
    "left": [0, -1]
}

# Utilizzo della funzione per ogni direzione
up = check_direction(grid, pos, offsets["up"])
down = check_direction(grid, pos, offsets["down"])
right = check_direction(grid, pos, offsets["right"])
left = check_direction(grid, pos, offsets["left"])

print(f"Up: {up}, Down: {down}, right: {right}, left: {left}")

