from gymnasium.envs.registration import register

register(
    id = "MARL_Cov-v0",
    entry_point= "MARL_Coverage.map:GridCoverage",
    max_episode_steps=30
)