# Function for making vector enviroment
import gymnasium as gym

def make_env(gym_id: str, n_agent: int) -> gym.spaces:
    def alias():
        env = gym.make(gym_id, n_agent = n_agent, map_id =1)
        return env
    return alias

