"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""

import gym
from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent
import gym_comm
import argparse

import sys
import json
from datetime import datetime

# TODO This is super hacky, but it works. There are some weird bugs with how gym_comm is imported in relation to gym_cooking, not sure how to fix this. 
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_dir, 'gym_cooking')
sys.path.append(relative_path)
# =============================

def create_arglist():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    parser.add_argument('--env-config',
                    type=json.loads,
                    default={},
                    help='Config for the environment')
    
    # parser.add_argument('--ego-save',
    #                     help='File to save the ego agent into')
    
    # parser.add_argument('--alt-save',
    #                     help='File to save the partner agent into')
    
    parser.add_argument('--total-timesteps', '-t',
                        type=int,
                        default=500000,
                        help='Number of time steps to run (ego perspective)')
    
    return parser.parse_args()

args = create_arglist()

env = gym.make('OvercookedMultiCommEnv-v0', **args.env_config)
partner = OnPolicyAgent(PPO('MultiInputPolicy', env, verbose=1))
env.add_partner_agent(partner)

# Finally, you can construct an ego agent and train it in the environment
ego = PPO('MultiInputPolicy', env, verbose=1)
ego.learn(total_timesteps=args.total_timesteps)

# Get the current time
current_time = datetime.now()

# Format the current time for file name
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S") 

ego.save(f"model/{args.env_config['level']}/{formatted_time}/ppo_ego")
partner.model.save(f"model/{args.env_config['level']}/{formatted_time}/ppo_partner1")