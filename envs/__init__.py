from gymnasium.envs.registration import register
from .coverage import GridCoverage

register(
    id = "MARL_Cov-v0",
    entry_point = "GridCoverage",
    max_episode_steps = 30
)