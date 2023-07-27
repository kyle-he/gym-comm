#!/bin/bash

# Declare the 'levels' array without spaces around the equal sign and without single quotes around array elements
all_levels=("full-divider_salad" "partial-divider_salad" "open-divider_salad" "full-divider_tomato" "partial-divider_tomato" "open-divider_tomato" "full-divider_tl" "partial-divider_tl" "open-divider_tl")
levels=("full-divider_salad" "open-divider_salad" "full-divider_tomato" "open-divider_tomato")

for level in "${levels[@]}"
do
    echo "Starting $level"
    python3 trainer.py --env-config "{\"level\": \"$level\", \"num_agents\": 2, \"max_num_timesteps\": 500}" -t 4000000 --log
    echo "Done with $level"
done
