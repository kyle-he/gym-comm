import gym
from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent
import gym_comm
import argparse

import sys
import json
from datetime import datetime
from gym_comm.extractors.CustomExtractor import CustomCombinedExtractor, FlattenedDictExtractor
from episode_recorder import EpisodeRecorder

import time
import csv
import traceback
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import DummyVecEnv
from vec_normalize import VecNormalize

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

    parser.add_argument('--hyperparams',
                       type=json.loads,
                       default={},
                       help='Hyperparameters for the training')
    
    # parser.add_argument('--ego-save',
    #                     help='File to save the ego agent into')
    
    # parser.add_argument('--alt-save',
    #                     help='File to save the partner agent into')
    
    parser.add_argument('--total-timesteps', '-t',
                        type=int,
                        default=500000,
                        help='Number of time steps to run (ego perspective)')
    
    parser.add_argument('--record-interval',
                        type=int,
                        default=-1,
                        help='Number of episodes to record. -1 to disable')

    parser.add_argument('--log',
                    action='store_true',
                    help='Log the run to runs/runlist.csv')
    
    parser.add_argument('--notes',
                        help='Notes to add to the run log')

    parser.add_argument('--wandb',
                        action='store_true',
                        help='Log the run to wandb')
    
    return parser.parse_args()

def start_training(args=None):
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    myenv = gym.make('OvercookedMultiCommEnv-v0', **args.env_config)
    # myenv = DummyVecEnv([lambda: myenv])  # The lambda function is used to make sure the environment is created in each subprocess
    # myenv = VecNormalize(venv=myenv, norm_obs_keys=["blockworld_map"])

    env = EpisodeRecorder(env=myenv, record_interval=args.record_interval)
    partner = OnPolicyAgent(PPO('MultiInputPolicy', env, 
                                n_steps = args.hyperparams.get('n_steps', 1000*5), 
                                batch_size=args.hyperparams.get('batch_size', 1000), 
                                ent_coef=args.hyperparams.get('entrop_coef', 0.01),
                                clip_range=args.hyperparams.get('clip_range', 0.05),
                                use_sde = args.hyperparams.get('sde', False),
                                learning_rate = args.hyperparams.get('learning_rate', 0.0003), 
                                policy_kwargs=policy_kwargs, verbose=1))
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

    ego_path = f"model/{args.env_config['level']}/{formatted_time}/ppo_ego"
    partner_path = f"model/{args.env_config['level']}/{formatted_time}/ppo_partner1"

    ego.save(ego_path)
    partner.model.save(partner_path)

    return ego_path, partner_path

if __name__ == "__main__":
    args = create_arglist()
    try:
        if args.wandb:
            wandb.login()
            run = wandb.init(
                # set the wandb project where this run will be logged
                project="gym_comm",

                name=f"{args.env_config['level']}_{args.total_timesteps}_learningrate_{args.hyperparams.get('learning_rate', 0.0003)}_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                
                # track hyperparameters and run metadata
                config={
                    "level": args.env_config['level'],
                    "num_agents": args.env_config['num_agents'],
                    "max_num_timesteps": args.env_config['max_num_timesteps'],
                    "total_timesteps": args.total_timesteps,
                },

                sync_tensorboard=True,

                save_code=True
            )

        start_time = time.time()

        ego_path, partner_path = start_training(args)

        run.finish()

        execution_time = time.time() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        log_run(args, ego_path=ego_path, partner_path=partner_path, successful=True, notes=f"Runtype: {args.notes}, Runtime: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")
    except Exception as error:
        log_run(args, error=error, successful=False)
        print("An error has occured: ")
        traceback_str = traceback.format_exc()
        print(traceback_str)