import gym
from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent
import gym_comm
import argparse

import sys
import json
from datetime import datetime
from gym_comm.extractors.CustomExtractor import CustomCombinedExtractor
from pantheonrl.common.agents import OnPolicyAgent, StaticPolicyAgent
import numpy as np

import time

# TODO This is super hacky, but it works. There are some weird bugs with how gym_comm is imported in relation to gym_cooking, not sure how to fix this. 
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_dir, 'gym_cooking')
sys.path.append(relative_path)
# =============================

class EnvException(Exception):
    """ Raise when parameters do not align with environment """

def create_arglist():
    parser = argparse.ArgumentParser("Overcooked 2 - Tester Argument parser")

    parser.add_argument('--env-config',
                    type=json.loads,
                    default={},
                    help='Config for the environment')
    
    parser.add_argument('--ego-load',
                        help='File to load the ego agent from')
    
    parser.add_argument('--alt-load',
                        help='File to load the partner agent from')
    
    parser.add_argument('--total-episodes', '-t',
                        type=int,
                        default=100,
                        help='Number of episodes to run')
    
    parser.add_argument('--render',
                        action='store_true',
                        help='Render the environment as it is being run')
    
    return parser.parse_args()

def gen_fixed(policy_type, location):
    agent = gen_load(policy_type, location)
    return StaticPolicyAgent(agent.policy)

def gen_load(policy_type, location):
    if policy_type == 'PPO':
        agent = PPO.load(location)
    else:
        raise EnvException("Not a valid FIXED/LOAD policy")

    return agent

def run_test(ego, env, num_episodes, render=False):
    env.set_ego_extractor(lambda obs: obs)
    rewards = []
    for game in range(num_episodes):
        obs = env.reset()
        done = False
        reward = 0
        if render:
            env.render()
        while not done:
            time.sleep (0.1)
            action = ego.get_action(obs, False)
            obs, newreward, done, _ = env.step(action)
            reward += newreward
            # print("Current Reward: ", newreward)

            # if render:
            #     env.render()
                # sleep(1/60)

        rewards.append(reward)
        print("------------------ Episode Complete ------------------")
        print("All Subtasks: ", env.base_env.all_subtasks)
        print("Completed Subtasks: ")
        for i, complete in enumerate(env.base_env.completed_subtasks):
            if complete == 1:
                print(f"{env.base_env.all_subtasks[i]}")
        print("Total Reward: ", reward)
        print(str(env.base_env))
        # import pdb; pdb.set_trace()

    env.close()
    print(f"Average Reward: {sum(rewards)/num_episodes}")
    print(f"Standard Deviation: {np.std(rewards)}")

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
)

args = create_arglist()
print(f"Arguments: {args}")

env = gym.make('OvercookedMultiCommEnv-v0', **args.env_config)
altenv = env.getDummyEnv(1)
print(f"Environment: {env}; Alt Environment: {altenv}")

ego = gen_fixed('PPO', args.ego_load)
print(f'Ego: {ego}')
alt = gen_fixed('PPO', args.alt_load)
env.add_partner_agent(alt)
print(f'Alt: {alt}')

run_test(ego, env, args.total_episodes, args.render)