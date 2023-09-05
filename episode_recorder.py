import gym
from datetime import datetime
from gym_cooking.misc.game.gameimage import GameImage

import sys
import wandb

class EpisodeRecorder(gym.Wrapper):
    def __init__(self, env, record_interval=-1):
        super(EpisodeRecorder, self).__init__(env)
        self.episode_count = 0
        self.record_interval = record_interval
    
    def step(self, action):
        observation, reward, done, info = super(EpisodeRecorder, self).step(action)

        self.env.base_env.display()
        if self.record_interval > 0 and self.episode_count % self.record_interval == 0:
            self.game.save_image_obs(self.env.base_env.t)
            if done:
                video_path = self.game.save_video()
                print("Saved video to {}".format(video_path))
                wandb.log({"gameplays": wandb.Video(video_path, caption=f"episode: {self.episode_count}", fps=5, format="mp4"), "step": self.episode_count})

        return observation, reward, done, info

    def reset(self, **kwargs):
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
            self.game.save_image_obs(self.env.base_env.t)
        
        return super_return