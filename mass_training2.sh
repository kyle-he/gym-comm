#!/bin/bash

# Declare the 'levels' array without spaces around the equal sign and without single quotes around array elements
all_levels=("open-divider_tomato" "full-divider_tomato")

for level in "${all_levels[@]}"
do
    echo "Starting $level"
    python3 trainer.py --env-config "{\"level\": \"$level\", \"num_agents\": 2, \"max_num_timesteps\": 500}" --hyperparams "{\"n_steps\": 2048, \"batch_size\": 512, \"entrop_coef\": 0.05}" -t 10000000 --notes "observation 1, episode_length = 300, record interval = 100, 3mil timesteps" --record-interval 100 --log --wandb
    echo "Done with $level"
done