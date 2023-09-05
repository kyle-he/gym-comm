import pygame
import os
import numpy as np
from PIL import Image
# from gym_cooking.misc.game.game import Game
from .game import Game
# from utils import *

import imageio

class GameImage(Game):
    def __init__(self, filename, world, sim_agents, record=False):
        Game.__init__(self, world, sim_agents)
        self.game_record_dir = 'gym_cooking/misc/game/record/{}'.format(filename)
        self.image_paths = []
        self.record = record

    def on_init(self):
        super().on_init()

        if self.record:
            # Make game_record folder if doesn't already exist
            if not os.path.exists(self.game_record_dir):
                os.makedirs(self.game_record_dir)

            # Clear game_record folder
            for f in os.listdir(self.game_record_dir):
                os.remove(os.path.join(self.game_record_dir, f))

    def get_image_obs(self):
        self.on_render()
        img_int = pygame.PixelArray(self.screen)
        img_rgb = np.zeros([img_int.shape[1], img_int.shape[0], 3], dtype=np.uint8)
        for i in range(img_int.shape[0]):
            for j in range(img_int.shape[1]):
                color = pygame.Color(img_int[i][j])
                img_rgb[j, i, 0] = color.g
                img_rgb[j, i, 1] = color.b
                img_rgb[j, i, 2] = color.r
        return img_rgb

    def save_image_obs(self, t):
        self.on_render()
        image_path = '{}/t={:03d}.png'.format(self.game_record_dir, t)
        pygame.image.save(self.screen, image_path)
        self.image_paths.append(image_path)
    
    def save_video(self):
        # Set the output video file
        output_video = '{}/output.mp4'.format(self.game_record_dir)

        # Create a writer object
        writer = imageio.get_writer(output_video, fps=5)

        # Iterate through the image filenames and add them to the video
        for filename in self.image_paths:
            image = imageio.imread(filename)
            writer.append_data(image)

        writer.close()

        return output_video
