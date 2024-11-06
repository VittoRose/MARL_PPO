# Function for making vector enviroment
import gymnasium as gym

def make_env(gym_id: str, idx: int, rnd: bool = False) -> gym.spaces:
    def alias():
        # If enviroment need more args add them here
        env = gym.make(gym_id)
        return env
    return alias

