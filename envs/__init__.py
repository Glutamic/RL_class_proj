# environments/__init__.py
from .maddpg import SimpleTagEnv

# Optional: If you want to register with PettingZoo/Gymnasium upon import
# from pettingzoo.utils import register_env as pettingzoo_register_env
#
# pettingzoo_register_env(
#     id='SimpleTagCustom-v0',
#     entry_point='marl_pursuit_evasion.environments:SimpleTagEnv'
# )
# Note: RLlib has its own registration mechanism which is often preferred
# when working primarily with RLlib.