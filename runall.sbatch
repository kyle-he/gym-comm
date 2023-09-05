#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --nodelist=ink-ron
#SBATCH --time=1440

# Declare the 'levels' array without spaces around the equal sign and without single quotes around array elements
levels=("partial-divider_tomato" "open-divider_tomato" "full-divider_tl" "partial-divider_tl" "open-divider_tl")
all_levels=("full-divider_salad" "partial-divider_salad" "open-divider_salad" "full-divider_tomato" "partial-divider_tomato" "open-divider_tomato" "full-divider_tl" "partial-divider_tl" "open-divider_tl")

for level in "${all_levels[@]}"
do
    echo "Starting $level"
    srun python3 trainer.py --env-config "{\"level\": \"$level\", \"num_agents\": 2, \"max_num_timesteps\": 500}" -t 4000000 --log &
    echo "Done with $level"
done

# Wait for all background jobs to complete
wait
