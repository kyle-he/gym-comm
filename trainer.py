import gym
from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent
import gym_comm
import argparse

import sys
import json
from datetime import datetime
from gym_comm.extractors.CustomExtractor import CustomCombinedExtractor, FlattenedDictExtractor
from episode_recorder import EpisodeRecorder, ParallelEpisodeRecorder

import time
import csv
import traceback
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from arglist import create_arglist

import os
import shutil
import json

# TODO This is super hacky, but it works. There are some weird bugs with how gym_comm is imported in relation to gym_cooking, not sure how to fix this. 
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_dir, 'gym_cooking')
sys.path.append(relative_path)
# =============================

# def log_run(args, ego_path=None, partner_path=None, error=None, successful=True, notes="None"):
#     if not args.log:
#         return

#     with open('runs/runlist.csv', 'a', newline='') as file:
#         writer = csv.writer(file)

#         headers = ['time', 'level', 'run_status', 'error', 'tester_command', 'num_agents', 'max_num_timesteps', 'total_timesteps', 'ego_location', 'alt_location', 'notes']
#         if file.tell() == 0:
#             writer.writerow(headers)

#         formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         data = [
#             formatted_time,
#             args.env_config['level'],
#             'SUCCESS' if successful else 'FAILURE',
#             error,
#             f'python3 tester.py --env-config \'{json.dumps(args.env_config)}\' --ego-load {ego_path} --alt-load {partner_path}',
#             args.env_config['num_agents'],
#             args.env_config['max_num_timesteps'],
#             args.total_timesteps,
#             ego_path,
#             partner_path,
#             notes,
#         ]
#         writer.writerow(data)

def make_args_from_json():
    parser = argparse.ArgumentParser("Overcooked 2 - Argument Parser")

    parser.add_argument('--json-path', '-j',
                        type=str,
                        default='env_args.json',
                        help='Path to the json file containing the arguments')

    args = parser.parse_args()
    json_path = args.json_path

    return json_path, create_arglist(args.json_path)

def start_training(args=None):
    policy_kwargs = dict(
        features_extractor_class=FlattenedDictExtractor
    )

    def make_overcooked_env():
        return EpisodeRecorder(env=gym.make('OvercookedMultiCommEnv-v0', arglist=args), record_interval=args.record_interval)
    
    # myenv = make_vec_env(make_overcooked_env, n_envs=4)
    # myenv = DummyVecEnv([lambda: myenv])  # The lambda function is used to make sure the environment is created in each subprocess
    # myenv = VecNormalize(venv=myenv, norm_obs_keys=["blockworld_map"])

    env = EpisodeRecorder(env=gym.make('OvercookedMultiCommEnv-v0', arglist=args), record_interval=args.record_interval)
    partner = OnPolicyAgent(PPO('MultiInputPolicy', env, 
                                n_steps = args.hyperparams.get('n_steps', 1000*5), 
                                batch_size=args.hyperparams.get('batch_size', 1000), 
                                ent_coef=args.hyperparams.get('entrop_coef', 0.01),
                                clip_range=args.hyperparams.get('clip_range', 0.05),
                                use_sde = args.hyperparams.get('sde', False),
                                learning_rate = args.hyperparams.get('learning_rate', 0.0003), 
                                policy_kwargs=policy_kwargs, verbose=1))

    # for env in env.envs:
    env.add_partner_agent(partner)

    # Finally, you can construct an ego agent and train it in the environment
    # TODO update clip range, noise?, maybe increase batch size = 512? or n_steps, adjust n_steps = 300*12
    ego = PPO('MultiInputPolicy', env, 
              n_steps = args.hyperparams.get('n_steps', 1000*5), 
              batch_size=args.hyperparams.get('batch_size', 1000), 
              ent_coef=args.hyperparams.get('entrop_coef', 0.01), 
              clip_range=args.hyperparams.get('clip_range', 0.05),
              learning_rate = args.hyperparams.get('learning_rate', 0.0003), 
              policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="runs")
    
    if args.wandb:
        ego.learn(total_timesteps=args.total_timesteps, callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ))
    else:
        ego.learn(total_timesteps=args.total_timesteps)

    # Get the current time
    current_time = datetime.now()

    # Format the current time for file name
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    ego_path = f"model/{args.level}/{run.id}_{formatted_time}/ppo_ego"1x
    partner_path = f"model/{args.level['level']}/{run.id}_{formatted_time}/ppo_partner1"

    ego.save(ego_path)
    partner.model.save(partner_path)

    return ego_path, partner_path

if __name__ == "__main__":
    json_path, args = make_args_from_json()
    try:
        if args.wandb:
            wandb.login()
            run = wandb.init(
                # set the wandb project where this run will be logged
                project="gym-communication-spread",

                name=f"{args.level}_{args.total_timesteps}_learningrate_{args.hyperparams.get('learning_rate', 0.0003)}_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                
                # track hyperparameters and run metadata
                config={
                    "level": args.level,
                    "num_agents": args.num_agents,
                    "max_num_timesteps": args.max_num_timesteps,
                    "total_timesteps": args.total_timesteps,
                },

                sync_tensorboard=True,

                save_code=True
            )
            shutil.copy(json_path, f'params/{run.id}_params.json')

        start_time = time.time()

        ego_path, partner_path = start_training(args)

        run.finish()

        execution_time = time.time() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # log_run(args, ego_path=ego_path, partner_path=partner_path, successful=True, notes=f"Runtype: {args.notes}, Runtime: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")
    except Exception as error:
        # log_run(args, error=error, successful=False)
        print("An error has occured: ")
        traceback_str = traceback.format_exc()
        print(traceback_str)