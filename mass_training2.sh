#!/bin/bash

# Declare the 'levels' array without spaces around the equal sign and without single quotes around array elements
all_levels=("partial-divider_tomato" "open-divider_tomato" "full-divider_tl" "partial-divider_tl" "open-divider_tl")

for level in "${all_levels[@]}"
do
    echo "Starting $level"
    python3 trainer.py --env-config "{\"level\": \"$level\", \"num_agents\": 2, \"max_num_timesteps\": 500}" -t 1000000 --log
    echo "Done with $level"
done
