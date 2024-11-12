# Container for reward value
rewards = {
    "still": -2,                    # Agent stay still
    "contact": -5,                  # Agent move into a wall
    "collision": -10,               # Agent bump into other agent
    "out": -5,                      # Agent move out map
    "tile_covered": -1,             # Agent moved in a previous covered tile
    "tile_not_covered": 10,         # Agent move in a empty tile
    "all_covered": 100,             # Agents covered all tiles (shared)
    "null": 0
}

# Each reward corrispond to a index in the table
reward_code = {
    "null" : 0,
    "contact": 1,
    "tile_not_covered": 2,
    "tile_covered": 3,
    "out": 4,
    "collision": 5,
    "still": 6
}

# Get an index and return the corrisponding action string for 
reward_decoder = {
    0: "null",
    1: "contact",
    2: "tile_not_covered",
    3: "tile_covered",
    4: "out",
    5: "collision",
    6: "still"
}