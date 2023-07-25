## Installation
```
# Optionally create a conda environment
conda create --name gym-rl
conda activate gym-rl
```

Install dependencies for [gym_cooking](https://github.com/rosewang2008/gym-cooking) and [pantheonrl](https://github.com/Stanford-ILIAD/PantheonRL)
```
# Navigate into PantheonRL and install requirements
cd pantheonrl
python3 -m pip install -e .

# Navigate back to main
cd .. 

# Navigate into gym_cooking and install requirements
cd gym_cooking
python3 -m pip install -e .
```

And you should be all set!

## Training

Here is an example of a basic command to train a model. 

```
python3 trainer.py --env-config '{"level": "open-divider_tomato", "num_agents": 2, "max_num_timesteps": 500}' -t 500000 --log
```

To see more advanced usage, run
```
python3 trainer.py --help
```

To test this trained model, you can access `runs/runlist.csv`, where you will be given a list of previously trained models and the command to test them.