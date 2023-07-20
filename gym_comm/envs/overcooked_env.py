import gym

import argparse
from pantheonrl.common.multiagentenv import SimultaneousEnv
# from gym_cooking.envs.overcooked_environment import OvercookedEnv
from gym_cooking.envs import OvercookedEnvironment
from utils.world import World
import utils.core as Core
import numpy as np
import re
import time
import sys

def create_arglist(args=None):
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

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

    # return parser.parse_args(["--level", "partial-divider_salad", "--num-agents", "2", "--max-num-timesteps", "500"])
    if args:
        return parser.parse_args(['--level', args['level'], '--num-agents', str(args['num_agents']), '--max-num-timesteps', str(args['max_num_timesteps'])])
    else:
        return parser.parse_args()

class OvercookedMultiEnv(SimultaneousEnv):
    def __init__(self, level, num_agents, max_num_timesteps, ego_agent_idx=0, baselines=False):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        super(OvercookedMultiEnv, self).__init__()

        # DEFAULT_ENV_PARAMS = {
        #     "horizon": 400
        # }
        # rew_shaping_params = {
        #     "PLACEMENT_IN_POT_REW": 3,
        #     "DISH_PICKUP_REWARD": 3,
        #     "SOUP_PICKUP_REWARD": 5,
        #     "DISH_DISP_DISTANCE_REW": 0,
        #     "POT_DISTANCE_REW": 0,
        #     "SOUP_DISTANCE_REW": 0,
        # }

        # self.mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name, rew_shaping_params=rew_shaping_params)
        # mlp = MediumLevelPlanner.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)

        args = {
            'level': level,
            'num_agents': num_agents,
            'max_num_timesteps': max_num_timesteps
        }
        
        self.base_env = OvercookedEnvironment(create_arglist(args))

        if baselines: np.random.seed(0)

        # map observation array
        observation_space_array = np.array([[0 for i in range(self.base_env.world.width)] for j in range(self.base_env.world.height)])
        map_observation_space = gym.spaces.Box(low=0, high=8, shape=((1,) + observation_space_array.shape), dtype=observation_space_array.dtype)

        # 0 = nothing, 1 = object, 2 = chopped object
        object_observation_space_array = np.array([[[0 for i in range(self.base_env.world.width)] for j in range(self.base_env.world.height)] for k in range(Core.NUM_OBJECT_CHANNELS)])
        object_observation_space = gym.spaces.Box(low=0, high=2, shape=((1,) + object_observation_space_array.shape), dtype=object_observation_space_array.dtype)

        # completed subtasks
        num_tasks = len(self.base_env.run_recipes())
        completed_subtasks_observation = gym.spaces.MultiBinary(num_tasks)

        self.observation_space = gym.spaces.Dict({
            'blockworld_map': map_observation_space,
            'object_map': object_observation_space,
            'completed_subtasks': completed_subtasks_observation
        })

        # Create a Tuple space for the actions
        self.lA = len(World.NAV_ACTIONS)
        self.action_space = gym.spaces.Discrete(self.lA)

        # self.action_space  = gym.spaces.Discrete( self.lA )
        self.ego_agent_idx = ego_agent_idx

        self.multi_reset()

        print(self.get_observation())
    
    def get_observation(self):
        self.base_env.display()

        observation_array_objects =  np.array([[[0 for i in range(self.base_env.world.width)] for j in range(self.base_env.world.height)] for k in range(Core.NUM_OBJECT_CHANNELS)])
        observation_array_map = np.array([[0 for i in range(self.base_env.world.width)] for j in range(self.base_env.world.height)])

        objs = []
        for o in self.base_env.world.objects.values():
            objs += o
        for obj in objs:
            x, y = obj.location
            if isinstance(obj, Core.Object):
                for content in obj.contents:
                    if (isinstance(content, Core.Food)):
                        observation_array_objects[Core.get_object_channel(content)][x][y]= content.state_index + 1
                    else:
                        observation_array_objects[Core.get_object_channel(content)][x][y] = 1
            elif isinstance(obj, Core.GridSquare):
                observation_array_map[x][y] = obj.get_value()

        observations = {
            'blockworld_map': observation_array_map.reshape((1,) + observation_array_map.shape),
            'object_map': observation_array_objects.reshape((1,) + observation_array_objects.shape),
            'completed_subtasks': np.array(self.base_env.completed_subtasks)
        }

        # encode next task, when doing rl when observations are the same then the goal is the same
        # add finished taskID to observation once its done
        # show that agent is holding and item 

        # print(observation_array)
        # print(shape)
        return observations
    
    # def string_to_number(self, string):

    def cost_fn(self):
        return 1
    
    def multi_step(self, ego_action, alt_action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
            encoded as an int

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        ego_action, alt_action = World.NAV_ACTIONS[ego_action], World.NAV_ACTIONS[alt_action]
        # also add potential communication action that adds to ego observation (tell the ego action)

        # use series of maps? 

        action_dict = {}
        
        if self.ego_agent_idx == 0:
            action_dict["agent-0"] = ego_action 
            action_dict["agent-1"] = alt_action 
        else:
            action_dict["agent-1"] = ego_action 
            action_dict["agent-0"] = alt_action 

        # base env to show what is being communicated
        reward, done, _ = self.base_env.step(action_dict)

        reward -= self.cost_fn()

        return (self.get_observation(), self.get_observation()), (reward, reward), done, {} #info

    def multi_reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        repr_obs = self.get_observation()

        return (repr_obs, repr_obs)

    def render(self, mode='human', close=False):
        print(self.base_env.rep)
        pass
