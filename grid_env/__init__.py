from gymnasium.envs.registration import register
from .coverage import GridCoverage

register(
    id="GridCoverage-v0",              
    entry_point="grid_env.coverage:GridCoverage",
    max_episode_steps=35                        # Truncated became true after 35 step
    )