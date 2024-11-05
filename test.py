import gymnasium as gym
import envs  

env = gym.make("GridCoverage-v0", map_id=1)

state, _ = env.reset()