import gym

import argparse
from pantheonrl.common.multiagentenv import SimultaneousEnv
# from gym_cooking.envs.overcooked_environment import OvercookedEnv
from gym_cooking.envs import OvercookedEnvironment
from gym_cooking.utils.world import World
import gym_cooking.utils.core as Core
import numpy as np
import re
import time

import sys

COMMUNICATION_ON = False
EGO_LED = False
FOW_RADIUS = 2

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
        arglist = ['--level', args['level'], '--num-agents', str(args['num_agents']), '--max-num-timesteps', str(args['max_num_timesteps'])]
        # import pdb; pdb.set_trace()
        return parser.parse_args(arglist)
    else:
        return parser.parse_args()

class OvercookedMultiEnv(SimultaneousEnv):
    def __init__(self, level, num_agents, max_num_timesteps, 
                 ego_agent_idx=0, baselines=False, num_communication=5):
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
            'max_num_timesteps': max_num_timesteps,
            'play': True
        }
        
        self.base_env = OvercookedEnvironment(create_arglist(args))

        if baselines: np.random.seed(0)

        # map observation array
        # TODO look at the dimensions of map, why remove reshape (1, ) +
        # observation_space_array = np.array([[0 for i in range(self.base_env.world.width)] for j in range(self.base_env.world.height)])
        # map_observation_space = gym.spaces.Box(low=0, high=8, shape=((1,) + observation_space_array.shape), dtype=observation_space_array.dtype)
        # map_observation_space = gym.spaces.Box(low=0, high=8, shape=(observation_space_array.shape), dtype=observation_space_array.dtype)

        # 0 = nothing, 1 = object, 2 = chopped object
        map_observation_space_array = np.array([[[0 for i in range(self.base_env.world.width)] for j in range(self.base_env.world.height)] for k in range(Core.NUM_OBJECT_CHANNELS + 2)])
        # map_observation_space_array = np.array([[0 for i in range(self.base_env.world.width)] for j in range(self.base_env.world.height)])
        
        # TODO update if changed
        map_observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(map_observation_space_array.shape), dtype=np.float32)

        agent_location_observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.base_env.world.width - 1, self.base_env.world.height - 1]), dtype=np.float32)
        # agent_location_observation_space = gym.spaces.MultiDiscrete([self.base_env.world.width, self.base_env.world.height])

        # completed subtasks
        num_tasks = len(self.base_env.run_recipes())
        completed_subtasks_observation = gym.spaces.MultiBinary(num_tasks)

        agent_holding_observation = gym.spaces.MultiBinary(2)

        # object observation space
        single_coordinate_space = gym.spaces.Tuple((gym.spaces.Discrete(8 * 2 + 1), gym.spaces.Discrete(8 * 2 + 1)))
        observation_space = gym.spaces.Tuple([single_coordinate_space for _ in range(4)])

        # self.observation_space = gym.spaces.Dict({
        #     'blockworld_map': map_observation_space,
        #     'completed_subtasks': completed_subtasks_observation,
        #     'agent_is_holding': agent_holding_observation
        # })

        num_tuples = 4
        
        width = self.base_env.world.width
        height = self.base_env.world.height

        flattened_object_encoding_space_x = gym.spaces.Box(low= -1 * width, high = width, shape=(num_tuples,), dtype=np.int64)
        flattened_object_encoding_space_y = gym.spaces.Box(low = height, high = height, shape=(num_tuples,), dtype=np.int64)
        
        object_state_space = gym.spaces.MultiBinary(4)

        self.observation_space = gym.spaces.Dict({
            'timestep': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'object_encodings_x': flattened_object_encoding_space_x,
            'object_encodings_y': flattened_object_encoding_space_y,
            'state_encodings': object_state_space,
            'is_hidden': object_state_space,
            'completed_subtasks': completed_subtasks_observation,
            'agent1_location': agent_location_observation_space,
            'agent2_location': agent_location_observation_space,
            'agent_is_holding': agent_holding_observation,
            'agent1_comm': gym.spaces.MultiBinary(num_communication),
            'agent2_comm': gym.spaces.MultiBinary(num_communication)
        })

        # Create a Tuple space for the actions
        self.lA = len(World.NAV_ACTIONS)
        self.action_space = gym.spaces.Discrete(self.lA)

        # multi-discrete space for communication
        self.action_space = gym.spaces.MultiDiscrete([self.lA, num_communication])

        self.num_communication = num_communication
        no_comm = np.zeros(self.num_communication)
        no_comm[0] = 1
        self.per_agent_communications = [no_comm for _ in range(num_agents)]

        # self.action_space  = gym.spaces.Discrete( self.lA )
        self.ego_agent_idx = ego_agent_idx

        self.multi_reset()

        # print(self.get_observation())
        print(self.get_observation2(0))
        print(self.get_observation2(1))

        # print(self.get_partial_observability_FOW(0))
        # print(self.get_partial_observability_FOW(1))
    
    def get_observation2(self, agent_idx, radius=1000):
        self.base_env.display()

        # for this case, we are garunteeing that the object will ALWAYS exist.
        object_distances = [(0, 0) for x in range(4)]
        object_states = [0 for x in range(4)]
        objs = []

        # agent location in map
        agent_x, agent_y = self.base_env.sim_agents[agent_idx].location

        for o in self.base_env.world.objects.values():
            objs += o
        for obj in objs:
            x, y = obj.location
            if isinstance(obj, Core.Object):
                for content in obj.contents:
                     if (isinstance(content, Core.Food)):
                         object_states[Core.get_object_channel(content)] = content.state_index
                     delta_x = x - agent_x
                     delta_y = y - agent_y
                     object_distances[Core.get_object_channel(content)] = (delta_x, delta_y)

        # M = 2 # 0 to 2
        # MAX_FLATTENED_VALUE = 3 ** 7 - 1
        # observation_array_flattened =  np.array([[0 for i in range(self.base_env.world.width)] for j in range(self.base_env.world.height)], dtype=np.float32)
        # for i in range(self.base_env.world.width):
        #     for j in range(self.base_env.world.height):
        #         combined_value = 0

        #         for k in range(Core.NUM_OBJECT_CHANNELS + 2):
        #             combined_value += (observation_array[k][i][j] * ((M + 1) ** (Core.NUM_OBJECT_CHANNELS + 1 - k)) / MAX_FLATTENED_VALUE)
                
        #         observation_array_flattened[i][j] = combined_value

        is_hidden = [0 if abs(x) + abs(y) <= radius else 1 for x, y in object_distances]
        visible_distances = [(0, 0) if abs(x) + abs(y) <= radius else (x, y) for x, y in object_distances]
        x_distances = list(pair[0] for pair in visible_distances)
        y_distances = list(pair[1] for pair in visible_distances)

        # agent_is_holding = []
        # for idx in range(0, 2):
        #     x, y = self.base_env.sim_agents[idx].location
        #     delta_x = agent_x - x
        #     delta_y = agent_y - y
        #     if abs(delta_x) + abs(delta_x) <= radius:
        #         agent_is_holding.append(self.base_env.sim_agents[idx].holding)
        #     else:
        #         agent_is_holding.append(-1)

        observations = {
            "timestep": np.array((self.base_env.t / self.base_env.arglist.max_num_timesteps, )),
            'object_encodings_x': np.array(x_distances),
            'object_encodings_y': np.array(y_distances),
            'state_encodings': np.array(object_states),
            'is_hidden': np.array(is_hidden),
            'completed_subtasks': np.array(self.base_env.completed_subtasks),
            'agent1_location': np.array(self.base_env.sim_agents[0].location),
            'agent2_location': np.array(self.base_env.sim_agents[1].location),
            'agent_is_holding': np.array((self.base_env.sim_agents[agent_idx].holding != None, False)),
            'agent1_comm': np.array(self.per_agent_communications[0]),
            'agent2_comm': np.array(self.per_agent_communications[1])
        }

        return observations

    # def get_observation(self):
    #     self.base_env.display()

    #     observation_array =  np.array([[[0 for i in range(self.base_env.world.width)] for j in range(self.base_env.world.height)] for k in range(Core.NUM_OBJECT_CHANNELS + 2)], dtype=np.float32)
    #     objs = []
    #     for o in self.base_env.world.objects.values():
    #         objs += o
    #     for obj in objs:
    #         x, y = obj.location
    #         if isinstance(obj, Core.Object):
    #             for content in obj.contents:
    #                 if (isinstance(content, Core.Food)):
    #                     observation_array[Core.get_object_channel(content) + 2][y][x]= (content.state_index + 1) / 2
    #                 else:
    #                     observation_array[Core.get_object_channel(content) + 2][y][x] = 1
    #         # elif isinstance(obj, Core.GridSquare):
    #         #     observation_array[0][x][y] = obj.get_obs_rep() / 3
        
    #     # encode agent locations
    #     for i, agent in enumerate(self.base_env.sim_agents):
    #         x, y = agent.location
    #         observation_array[i][y][x] = 1

    #     # M = 2 # 0 to 2
    #     # MAX_FLATTENED_VALUE = 3 ** 7 - 1
    #     # observation_array_flattened =  np.array([[0 for i in range(self.base_env.world.width)] for j in range(self.base_env.world.height)], dtype=np.float32)
    #     # for i in range(self.base_env.world.width):
    #     #     for j in range(self.base_env.world.height):
    #     #         combined_value = 0

    #     #         for k in range(Core.NUM_OBJECT_CHANNELS + 2):
    #     #             combined_value += (observation_array[k][i][j] * ((M + 1) ** (Core.NUM_OBJECT_CHANNELS + 1 - k)) / MAX_FLATTENED_VALUE)
                
    #     #         observation_array_flattened[i][j] = combined_value

    #     observations = {
    #         'blockworld_map': observation_array,
    #         'completed_subtasks': np.array(self.base_env.completed_subtasks),
    #         'agent_is_holding': np.array(((self.base_env.sim_agents[0].holding != None), (self.base_env.sim_agents[1].holding != None)))
    #     }

    #     return observations
    
    def get_partial_observability_FOW(self, agent_idx, radius=2):
        """
        Returns the partial observability field of view for the agent with index agent_idx for a fog of war scenario
        """
        self.base_env.display()

        observation_array =  np.array([[[0 for i in range(self.base_env.world.width)] for j in range(self.base_env.world.height)] for k in range(Core.NUM_OBJECT_CHANNELS + 3)])
        objs = []
        for o in self.base_env.world.objects.values():
            objs += o
        for obj in objs:
            x, y = obj.location
            if isinstance(obj, Core.Object):
                for content in obj.contents:
                    if (isinstance(content, Core.Food)):
                        observation_array[Core.get_object_channel(content) + 3][x][y]= content.state_index + 1
                    else:
                        observation_array[Core.get_object_channel(content) + 3][x][y] = 1
            elif isinstance(obj, Core.GridSquare):
                observation_array[0][x][y] = obj.get_obs_rep()
        
        # encode agent locations
        for i, agent in enumerate(self.base_env.sim_agents):
            x, y = agent.location
            observation_array[i + 1][x][y] = 1

        for x in range(self.base_env.world.width):
            for y in range(self.base_env.world.height):
                agent_x, agent_y = self.base_env.sim_agents[agent_idx].location
                distance = abs(x - agent_x) + abs(y - agent_y)

                if distance > radius:
                    for k in range(Core.NUM_OBJECT_CHANNELS + 3):
                        observation_array[k][x][y] = -1

        observations = {
            'blockworld_map': observation_array.reshape(observation_array.shape),
            'completed_subtasks': np.array(self.base_env.completed_subtasks),
            'agent_is_holding': np.array(((self.base_env.sim_agents[0].holding != None), (self.base_env.sim_agents[1].holding != None)))
        }

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
        # ego_action, alt_action = World.NAV_ACTIONS[ego_action], World.NAV_ACTIONS[alt_action]
        # also add potential communication action that adds to ego observation (tell the ego action)

        # use series of maps? 

        action_dict = {}

        ego_action, ego_communication = ego_action
        alt_action, alt_communication = alt_action

        # print("EGO COMMUNICATION: ", ego_communication)
        # print("ALT COMMUNICATION: ", alt_communication)

        # let's convert this into a one-hot vector
        ego_communication = np.zeros(self.num_communication)

        if COMMUNICATION_ON:
            ego_communication[ego_action] = 1

        alt_communication = np.zeros(self.num_communication)
        if COMMUNICATION_ON:
            if not EGO_LED:
                alt_communication[alt_action] = 1

        self.per_agent_communications[0] = ego_communication
        self.per_agent_communications[1] = alt_communication

        ego_action, alt_action = World.NAV_ACTIONS[ego_action], World.NAV_ACTIONS[alt_action]
        
        if self.ego_agent_idx == 0:
            action_dict["agent-0"] = ego_action 
            action_dict["agent-1"] = alt_action
            # action_dict["agent-1"] = (0, 0)
        else:
            action_dict["agent-1"] = ego_action 
            action_dict["agent-0"] = alt_action 

        # base env to show what is being communicated
        reward, done, info = self.base_env.step(action_dict)

        # reward -= self.cost_fn()
        if (reward != 0):

            print(str(self.base_env))

            print("==== EGO AGENT OBS ====")
            print(self.get_observation2(0, radius=FOW_RADIUS))

            print("==== PARTNER AGENT OBS ====")
            print(self.get_observation2(1, radius=FOW_RADIUS))

            print("Reward Value: ", reward)
            print("Agent 0 Reward Shaping: ", info["agent_0_reward_shaping"])
            print("Agent 1 Reward Shaping: ", info["agent_1_reward_shaping"])

            # print("Observations")
            # print(self.get_observation())

        # print(str(self.base_env))
        # print(self.get_observation())
        # print("Reward Value: ", reward)

        # shared_reward = reward - info["agent_0_reward_shaping"] - info["agent_1_reward_shaping"]

        # print(self.get_partial_observability_FOW(0))
        # print(self.get_partial_observability_FOW(1))
        # import pdb; pdb.set_trace()


        # return ((self.get_partial_observability_FOW(0), self.get_partial_observability_FOW(1)), (reward - info["agent_0_reward_shaping"], reward - info["agent_1_reward_shaping"]), done, {}) #info
        
        return (self.get_observation2(0, radius=FOW_RADIUS), self.get_observation2(1, radius=FOW_RADIUS)), (reward - info["agent_0_reward_shaping"], reward - info["agent_1_reward_shaping"]), done, {} #info

        # return ((self.get_observation(), self.get_observation()), (reward - info["agent_0_reward_shaping"], reward - info["agent_1_reward_shaping"]), done, {}) #info

        # return ((self.get_partial_observability_FOW(0), self.get_partial_observability_FOW(1)), (reward, reward), done, {}) #info

    def multi_reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        # repr_obs = self.get_observation()

        # return (self.get_observation(), self.get_observation())
        return (self.get_observation2(0, radius=FOW_RADIUS), self.get_observation2(1, radius=FOW_RADIUS))

    def render(self, mode='human', close=False):
        print(str(self.base_env))
        print(self.get_observation())
        pass
