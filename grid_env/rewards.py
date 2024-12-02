# Container for reward value
rewards = {
    "still": -0.1,                      # Agent stay still
    "contact": -0.5,                    # Agent move into a wall
    "collision": -0.8,                  # Agent bump into other agent
    "out": -0.5,                        # Agent move out map
    "tile_covered": -0.15,              # Agent moved in a previous covered tile
    "tile_not_covered": 1,              # Agent move in a empty tile
    "all_covered": 200,                 # Agents covered all tiles (shared)
    "null": 0
}

# Each reward correspond to a index in the table
reward_code = {
    "null" : 0,
    "contact": 1,
    "tile_not_covered": 2,
    "tile_covered": 3,
    "out": 4,
    "collision": 5,
    "still": 6
}

# Get an index and return the corresponding action string
reward_decoder = {
    0: "null",
    1: "contact",
    2: "tile_not_covered",
    3: "tile_covered",
    4: "out",
    5: "collision",
    6: "still"
}