import json
import argparse

def create_arglist(json_path):
    parser = argparse.ArgumentParser("Overcooked 2 - Parser")

    # parser.add_argument('--env-config',
    #                 type=json.loads,
    #                 default={},
    #                 help='Config for the environment')

    parser.add_argument('--hyperparams',
                       type=json.loads,
                       default={},
                       help='Hyperparameters for the training')
    
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

    #### BASE ENVIRONMENT PARAMS ####
    # Environment
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=100, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")

    # Delegation Planner
    parser.add_argument("--beta", type=float, default=1.3, help="Beta for softmax in Bayesian delegation updates")

    # Navigation Planner
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for BRTDP")
    parser.add_argument("--tau", type=int, default=2, help="Normalize v diff")
    parser.add_argument("--cap", type=int, default=75, help="Max number of steps in each main loop of BRTDP")
    parser.add_argument("--main-cap", type=int, default=100, help="Max number of main loops in each run of BRTDP")

    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")
    # Models
    # Valid options: `bd` = Bayes Delegation; `up` = Uniform Priors
    # `dc` = Divide & Conquer; `fb` = Fixed Beliefs; `greedy` = Greedy
    parser.add_argument("--model1", type=str, default=None, help="Model type for agent 1 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model2", type=str, default=None, help="Model type for agent 2 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model3", type=str, default=None, help="Model type for agent 3 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model4", type=str, default=None, help="Model type for agent 4 (bd, up, dc, fb, or greedy)")


    parser.add_argument('--communication-on',
                        action='store_true',
                        help='Enable communication')
    parser.add_argument('--num-communication',
                        type=int,
                        default=10,
                        help='Number of communication channels')
    parser.add_argument('--ego-led',
                        action='store_true',
                        help='Enable ego LED')
    parser.add_argument('--fow-radius',
                        type=int,
                        default=2,
                        help='Field of view radius')

    ## Agent Configs
    parser.add_argument('--ego-config',
                       type=json.loads,
                       default={},
                       help='Ego Config')
    
    parser.add_argument('--partner-config',
                       type=json.loads,
                       default={},
                       help='Partner Config')

    ### LOAD JSON FILE ###
    
    with open(json_path, 'r') as file:
        args = json.load(file)

    arglist = [
        '--level', args['level'], 
        '--num-agents', str(args['num_agents']), 
        '--max-num-timesteps', str(args['max_num_timesteps']),
        '--hyperparams', json.dumps(args['hyperparams']),
        '--ego-config', json.dumps(args['ego_config']),
        '--partner-config', json.dumps(args['partner_config']),
        '--total-timesteps', str(args.get('total_timesteps', 20000000)),  
        '--record-interval', str(args.get('record_interval', 500)),      
        '--log' if args.get('log', False) else '',
        '--notes', args.get('notes', 'XXX notes'),                  
        '--wandb' if args.get('wandb', False) else '',
        '--communication-on' if args.get('communication_on', False) else '',
        '--num-communication', str(args.get('num_communication', 10)),
        '--ego-led' if args.get('ego_led', False) else '',
        '--fow-radius', str(args.get('fow_radius', 2))
    ]

    arglist = [arg for arg in arglist if arg != '']

    print(arglist)

    return parser.parse_args(arglist)
