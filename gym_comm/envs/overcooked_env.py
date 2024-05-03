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

class OvercookedMultiEnv(SimultaneousEnv):
    def __init__(self, arglist,
                 ego_agent_idx=0, baselines=False):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        super(OvercookedMultiEnv, self).__init__()

        # args = {
        #     'level': level,
        #     'num_agents': num_agents,
        #     'max_num_timesteps': max_num_timesteps,
        #     'play': True
        # }

        self.arglist = arglist
        self.base_env = OvercookedEnvironment(arglist)

        if baselines: np.random.seed(0)

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
            'agent1_comm': gym.spaces.MultiBinary(self.arglist.num_communication),
            'agent2_comm': gym.spaces.MultiBinary(self.arglist.num_communication)
        })

        # Create a Tuple space for the actions
        self.lA = len(World.NAV_ACTIONS)
        self.action_space = gym.spaces.Discrete(self.lA)

        # multi-discrete space for communication
        self.action_space = gym.spaces.MultiDiscrete([self.lA, self.arglist.num_communication])

        print(f"Num Commmunication Channels: {self.arglist.num_communication}")

        no_comm = np.zeros(self.arglist.num_communication)
        no_comm[0] = 1
        self.per_agent_communications = [no_comm for _ in range(self.arglist.num_agents)]

        # self.action_space  = gym.spaces.Discrete( self.lA )
        self.ego_agent_idx = ego_agent_idx

        self.multi_reset()

        # print(self.get_observation())
        print(self.get_observation2(0))
        print(self.get_observation2(1))

        # print(self.get_partial_observability_FOW(0))
        # print(self.get_partial_observability_FOW(1))
    
    def get_observation2(self, agent_idx, radius=1000):
        # Initialize default values for observations
        object_distances = [(0, 0) for x in range(4)]
        object_states = [0 for x in range(4)]
        is_hidden = [1 for x in range(4)]  # Set to all 1's for BLIND condition
        objs = []

        # agent location in map
        agent_x, agent_y = self.base_env.sim_agents[agent_idx].location

        if agent_idx == 0:
            IS_BLIND = self.arglist.ego_config["BLIND"]
        else:
            IS_BLIND = self.arglist.partner_config["BLIND"]

        if not IS_BLIND:
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

            is_hidden = [0 if abs(x) + abs(y) <= radius else 1 for x, y in object_distances]

        visible_distances = [(0, 0) if abs(x) + abs(y) <= radius else (x, y) for x, y in object_distances]
        x_distances = list(pair[0] for pair in visible_distances)
        y_distances = list(pair[1] for pair in visible_distances)

        agent1_location = np.array([0, 0])
        agent2_location = np.array([0, 0])
        if not IS_BLIND:
            agent1_location = np.array(self.base_env.sim_agents[0].location)
            agent2_location = np.array(self.base_env.sim_agents[1].location)

        observations = {
            "timestep": np.array((self.base_env.t / self.base_env.arglist.max_num_timesteps, )),
            'object_encodings_x': np.array(x_distances),
            'object_encodings_y': np.array(y_distances),
            'state_encodings': np.array(object_states),
            'is_hidden': np.array(is_hidden),
            'completed_subtasks': np.array(self.base_env.completed_subtasks),
            'agent1_location': agent1_location,
            'agent2_location': agent2_location, 
            'agent_is_holding': np.array((0, 0)) if self.arglist.ego_config["BLIND"] else np.array((self.base_env.sim_agents[agent_idx].holding != None, False)),
            'agent1_comm': np.array(self.per_agent_communications[0]),
            'agent2_comm': np.array(self.per_agent_communications[1])
        }

        return observations
    
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

        action_dict = {}

        ego_action, ego_communication_val = ego_action
        alt_action, alt_communication_val = alt_action

        # print(f"EGO COMM: {ego_communication}")
        # print(f"ALT COMM: {alt_communication}")

        # let's convert this into a one-hot vector
        ego_communication = np.zeros(self.arglist.num_communication)

        if self.arglist.communication_on:
            ego_communication[ego_communication_val] = 1

        alt_communication = np.zeros(self.arglist.num_communication)
        if self.arglist.communication_on:
            if not self.arglist.ego_led:
                alt_communication[alt_communication_val] = 1
        # object_locations = self.get_object_locations()
        # binary_vector = np.zeros(30)  # Adjust size based on encoding needs

        # # Assuming each coordinate is represented with up to 9 bits
        # for i, obj in enumerate(['tomato', 'lettuce', 'plate']):
        #     x, y = object_locations[obj]
        #     binary_vector[i*9:(i*9)+4] = [int(b) for b in format(x, '04b')]  # 4 bits for x
        #     binary_vector[(i*9)+4:(i+1)*9] = [int(b) for b in format(y, '05b')]  # 5 bits for y

        self.per_agent_communications[0] = ego_communication
        self.per_agent_communications[1] = alt_communication

        ego_action, alt_action = World.NAV_ACTIONS[ego_action], World.NAV_ACTIONS[alt_action]

        action_dict["agent-1"] = (0, 0)
        action_dict["agent-0"] = (0, 0)

        if self.ego_agent_idx == 0:
            if self.arglist.ego_config["CAN_MOVE"]:
                action_dict["agent-0"] = ego_action 
            if self.arglist.partner_config["CAN_MOVE"]:
                action_dict["agent-1"] = alt_action
        else:
            if self.arglist.ego_config["CAN_MOVE"]:
                action_dict["agent-1"] = ego_action 
            if self.arglist.partner_config["CAN_MOVE"]:
                action_dict["agent-0"] = alt_action

        # base env to show what is being communicated
        reward, done, info = self.base_env.step(action_dict)

        # reward -= self.cost_fn()
        if (reward != 0):

            print(str(self.base_env))

            print("==== EGO AGENT OBS ====")
            print(self.get_observation2(0, radius=self.arglist.fow_radius))

            print("==== PARTNER AGENT OBS ====")
            print(self.get_observation2(1, radius=self.arglist.fow_radius))

            print("Reward Value: ", reward)
            print("Agent 0 Reward Shaping: ", info["agent_0_reward_shaping"])
            print("Agent 1 Reward Shaping: ", info["agent_1_reward_shaping"])

        return (self.get_observation2(0, radius=self.arglist.fow_radius), self.get_observation2(1, radius=self.arglist.fow_radius)), (reward - info["agent_0_reward_shaping"] - info["agent_1_reward_shaping"], reward - info["agent_0_reward_shaping"] - info["agent_1_reward_shaping"]), done, {} #info

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
        return (self.get_observation2(0, radius=self.arglist.fow_radius), self.get_observation2(1, radius=self.arglist.fow_radius))

    def render(self, mode='human', close=False):
        print(str(self.base_env))
        print(self.get_observation())
        pass
