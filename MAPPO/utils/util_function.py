import gymnasium as gym

def make_env(gym_id: str, n_agent: int, map_id: int) -> gym.spaces:
    def alias():
        env = gym.make(gym_id, n_agent = n_agent, map_id=map_id)
        return env
    return alias

