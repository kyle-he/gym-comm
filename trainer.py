import gym
from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent
import gym_comm
import argparse

import sys
import json
from datetime import datetime
from gym_comm.extractors.CustomExtractor import CustomCombinedExtractor

import time
import csv
import traceback

# TODO This is super hacky, but it works. There are some weird bugs with how gym_comm is imported in relation to gym_cooking, not sure how to fix this. 
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_dir, 'gym_cooking')
sys.path.append(relative_path)
# =============================

def log_run(args, ego_path=None, partner_path=None, error=None, successful=True, notes="None"):
    if not args.log:
        return

    with open('runs/runlist.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        headers = ['time', 'level', 'run_status', 'error', 'tester_command', 'num_agents', 'max_num_timesteps', 'total_timesteps', 'ego_location', 'alt_location', 'notes']
        if file.tell() == 0:
            writer.writerow(headers)

        formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data = [
            formatted_time,
            args.env_config['level'],
            'SUCCESS' if successful else 'FAILURE',
            error,
            f'python3 tester.py --env-config \'{json.dumps(args.env_config)}\' --ego-load {ego_path} --alt-load {partner_path}',
            args.env_config['num_agents'],
            args.env_config['max_num_timesteps'],
            args.total_timesteps,
            ego_path,
            partner_path,
            notes,
        ]
        writer.writerow(data)

def create_arglist():
    parser = argparse.ArgumentParser("Overcooked 2 - Trainer Argument parser")

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

    parser.add_argument('--log',
                    action='store_true',
                    help='Log the run to runs/runlist.csv')
    
    return parser.parse_args()

def start_training(args=None):
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    env = gym.make('OvercookedMultiCommEnv-v0', **args.env_config)
    partner = OnPolicyAgent(PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs, verbose=1))
    env.add_partner_agent(partner)

    # Finally, you can construct an ego agent and train it in the environment
    ego = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    ego.learn(total_timesteps=args.total_timesteps)

    # Get the current time
    current_time = datetime.now()

    # Format the current time for file name
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S") 

    ego_path = f"model/{args.env_config['level']}/{formatted_time}/ppo_ego"
    partner_path = f"model/{args.env_config['level']}/{formatted_time}/ppo_partner1"

    ego.save(ego_path)
    partner.model.save(partner_path)

    return ego_path, partner_path

if __name__ == "__main__":
    args = create_arglist()
    try:
        start_time = time.time()

        ego_path, partner_path = start_training(args)

        execution_time = time.time() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        log_run(args, ego_path=ego_path, partner_path=partner_path, successful=True, notes=f"Runtime: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")
    except Exception as error:
        log_run(args, error=error, successful=False)
        print("An error has occured: ")
        traceback_str = traceback.format_exc()
        print(traceback_str)