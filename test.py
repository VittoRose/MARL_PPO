import gymnasium as gym
import envs  

env = gym.make("GridCoverage-v0", n_agent=2, map_id=1)

state, _ = env.reset()

action = [0,0]

env.step(action)