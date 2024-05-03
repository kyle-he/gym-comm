#!/bin/bash
#SBATCH --job-name=2allergic_comm
#SBATCH --gres=gpu:1080:1
#SBATCH --qos=general
#SBATCH --time=3000
#SBATCH --array=1-6%6
#SBATCH --output=job_%A_%a.out

# Define an array of JSON file paths (using relative paths)
json_files=(
    "spread/env_args8off_allergic.json"
    "spread/env_args8on_allergic.json"
    "spread/env_args2off_allergic.json"
    "spread/env_args2on_allergic.json"
    "spread/env_args20off_allergic.json"
    "spread/env_args20on_allergic.json"
)

# Use the SLURM_ARRAY_TASK_ID to select the appropriate JSON file
json_path=${json_files[$((SLURM_ARRAY_TASK_ID - 1))]}

# Run the Python script with the selected JSON file
python3 trainer.py --json-path "$json_path"