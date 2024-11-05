from gymnasium.envs.registration import register
from .coverage import GridCoverage

register(
    id="GridCoverage-v0",              
    entry_point="envs.coverage:GridCoverage",  
)