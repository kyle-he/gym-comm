"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""

import gym
from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent
import gym_comm

import sys
from datetime import datetime

sys.path.append("/home/kyle/code/gymRL/gym_cooking")
# from overcookedgym.overcooked_utils import LAYOUT_LIST

# Since pantheonrl's MultiAgentEnv is a subclass of the gym Env, you can
# register an environment and construct it using gym.make.
env = gym.make('OvercookedMultiCommEnv-v0')

# Before training your ego agent, you first need to add your partner agents
# to the environment. You can create adaptive partner agents using
# OnPolicyAgent (for PPO/A2C) or OffPolicyAgent (for DQN/SAC). If you set
# verbose to true for these agents, you can also see their learning progress
partner = OnPolicyAgent(PPO('MultiInputPolicy', env, verbose=1))
env.add_partner_agent(partner)

# Finally, you can construct an ego agent and train it in the environment
ego = PPO('MultiInputPolicy', env, verbose=1)
ego.learn(total_timesteps=500000)

# Get the current time
current_time = datetime.now()

# Format the current time for file name
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S") 

ego.save(f"model/open-divider_tomato/{formatted_time}/ppo_ego_500")
partner.model.save(f"model/open-divider_tomato/{formatted_time}/ppo_partner1_500")