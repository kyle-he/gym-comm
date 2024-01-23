import gym
from datetime import datetime
from gym_cooking.misc.game.gameimage import GameImage

import sys
import wandb
import numpy as np

class EpisodeRecorder(gym.Wrapper):
    def __init__(self, env, record_interval=-1):
        super(EpisodeRecorder, self).__init__(env)
        self.episode_count = 0
        self.record_interval = record_interval
    
    def step(self, action):
        observation, reward, done, info = super(EpisodeRecorder, self).step(action)

        self.env.base_env.display()
        if self.record_interval > 0 and self.episode_count % self.record_interval == 0:
            self.game.save_image_obs(self.env.base_env.t, [f"Ego: {np.argmax(observation['agent1_comm'])}", f"Partner: {np.argmax(observation['agent2_comm'])}"])
            if done:
                video_path = self.game.save_video()
                print("Saved video to {}".format(video_path))
                wandb.log({"gameplays": wandb.Video(video_path, caption=f"episode: {self.episode_count}", fps=5, format="mp4"), "step": self.episode_count})

        return observation, reward, done, info

    def reset(self, **kwargs):
        wandb.log({"num_completed_subtasks": sum(self.env.base_env.completed_subtasks)}, step=self.episode_count)
        
        super_return = super(EpisodeRecorder, self).reset(**kwargs)

        self.episode_count += 1
        if self.record_interval > 0 and self.episode_count % self.record_interval == 0:
            formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.video_file_path = f"{self.env.base_env.arglist.level}/{formatted_time}"
            self.game = GameImage(
                    filename=f"{self.env.base_env.arglist.level}/{formatted_time}",
                    world=self.env.base_env.world,
                    sim_agents=self.env.base_env.sim_agents,
                    record=True)
            self.game.on_init()
            self.game.save_image_obs(self.env.base_env.t, ["Starting game"])
        
        return super_return


class ParallelEpisodeRecorder(gym.Wrapper):
    def __init__(self, env, record_interval=-1):
        super(ParallelEpisodeRecorder, self).__init__(env)
        self.num_envs = getattr(env, 'num_envs', 1)  # Get the number of parallel environments
        self.episode_counts = [0 for _ in range(self.num_envs)]
        self.record_interval = record_interval
        self.games = [None for _ in range(self.num_envs)]  # Initialize GameImage instances

    def step(self, actions):
        observations, rewards, dones, infos = super(ParallelEpisodeRecorder, self).step(actions)

        for i in range(self.num_envs):
            # Logic for each environment
            if self.record_interval > 0 and self.episode_counts[i] % self.record_interval == 0:
                self.games[i].save_image_obs(self.env.get_attr('t', i)[0])  # Assuming `t` is a time attribute in your env
                if dones[i]:
                    video_path = self.games[i].save_video()
                    print(f"Saved video for env {i} to {video_path}")
                    wandb.log({"gameplays": wandb.Video(video_path, caption=f"episode: {self.episode_counts[i]}", fps=5, format="mp4"), "step": self.episode_counts[i]})

        return observations, rewards, dones, infos

    def reset(self, **kwargs):
        super_returns = super(ParallelEpisodeRecorder, self).reset(**kwargs)

        for i in range(self.num_envs):
            self.episode_counts[i] += 1
            if self.record_interval > 0 and self.episode_counts[i] % self.record_interval == 0:
                formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                video_file_path = f"{self.env.get_attr('arglist', i)[0].level}/{formatted_time}"  # Assuming `arglist` and `level` attributes in your env
                self.games[i] = GameImage(
                    filename=video_file_path,
                    world=self.env.get_attr('world', i)[0],  # Assuming `world` attribute in your env
                    sim_agents=self.env.get_attr('sim_agents', i)[0],  # Assuming `sim_agents` attribute in your env
                    record=True)
                self.games[i].on_init()
                self.games[i].save_image_obs(self.env.get_attr('t', i)[0])

        return super_returns