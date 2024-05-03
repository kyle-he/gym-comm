# Recipe planning
from gym_cooking.recipe_planner.stripsworld import STRIPSWorld
import gym_cooking.recipe_planner.utils as recipe
from gym_cooking.recipe_planner.recipe import *

# Delegation planning
from gym_cooking.delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planning
import gym_cooking.navigation_planner.utils as nav_utils

# Other core modules
from gym_cooking.utils.interact import interact
from gym_cooking.utils.world import World
from gym_cooking.utils.core import *
from gym_cooking.utils.agent import SimAgent
# from gym_cooking.misc.game.gameimage import GameImage
from gym_cooking.utils.agent import COLORS

import copy
import networkx as nx
import numpy as np
from itertools import combinations, permutations, product
from collections import namedtuple
from datetime import datetime

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import sys
import wandb

CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")
import time

class OvercookedEnvironment(gym.Env):
    """Environment object for Overcooked."""

    def __init__(self, arglist):
        self.arglist = arglist
        self.t = 0
        self.set_filename()

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

        self.goal_objects_count = []
        self.completed_subtasks = []

        self.reset()
        self.display()

    def get_repr(self):
        return self.world.get_repr() + tuple([agent.get_repr() for agent in self.sim_agents])

    def __str__(self):
        # Print the world and agents.
        _display = list(map(lambda x: ''.join(map(lambda y: y + ' ', x)), self.rep))
        return '\n'.join(_display)

    def __eq__(self, other):
        return self.get_repr() == other.get_repr()

    def __copy__(self):
        new_env = OvercookedEnvironment(self.arglist)
        new_env.__dict__ = self.__dict__.copy()
        new_env.world = copy.copy(self.world)
        new_env.sim_agents = [copy.copy(a) for a in self.sim_agents]
        new_env.distances = self.distances

        # Make sure new objects and new agents' holdings have the right pointers.
        for a in new_env.sim_agents:
            if a.holding is not None:
                a.holding = new_env.world.get_object_at(
                        location=a.location,
                        desired_obj=None,
                        find_held_objects=True)
        return new_env

    def set_filename(self):
        self.filename = "{}_agents{}_seed{}".format(self.arglist.level,\
            self.arglist.num_agents, self.arglist.seed)
        model = ""
        if self.arglist.model1 is not None:
            model += "_model1-{}".format(self.arglist.model1)
        if self.arglist.model2 is not None:
            model += "_model2-{}".format(self.arglist.model2)
        if self.arglist.model3 is not None:
            model += "_model3-{}".format(self.arglist.model3)
        if self.arglist.model4 is not None:
            model += "_model4-{}".format(self.arglist.model4)
        self.filename += model

    def load_level(self, level, num_agents):
        x = 0
        y = 0
        with open('gym_cooking/utils/levels/{}.txt'.format(level), 'r') as file:
            # Mark the phases of reading.
            phase = 1
            for line in file:
                line = line.strip('\n')
                if line == '':
                    phase += 1

                # Phase 1: Read in kitchen map.
                elif phase == 1:
                    for x, rep in enumerate(line):
                        # Object, i.e. Tomato, Lettuce, Onion, or Plate.
                        if rep in 'tlop':
                            counter = Counter(location=(x, y))
                            obj = Object(
                                    location=(x, y),
                                    contents=RepToClass[rep]())
                            counter.acquire(obj=obj)
                            self.world.insert(obj=counter)
                            self.world.insert(obj=obj)
                        # GridSquare, i.e. Floor, Counter, Cutboard, Delivery.
                        elif rep in RepToClass:
                            newobj = RepToClass[rep]((x, y))
                            self.world.objects.setdefault(newobj.name, []).append(newobj)
                        else:
                            # Empty. Set a Floor tile.
                            f = Floor(location=(x, y))
                            self.world.objects.setdefault('Floor', []).append(f)
                    y += 1
                # Phase 2: Read in recipe list.
                elif phase == 2:
                    self.recipes.append(globals()[line]())

                # Phase 3: Read in agent locations (up to num_agents).
                elif phase == 3:
                    if len(self.sim_agents) < num_agents:
                        loc = line.split(' ')
                        if (len(self.sim_agents) == 0):
                            agent_config = self.arglist.ego_config
                        else:
                            agent_config = self.arglist.partner_config

                        sim_agent = SimAgent(
                                name='agent-'+str(len(self.sim_agents)),
                                id_color=COLORS[len(self.sim_agents)],
                                location=(int(loc[0]), int(loc[1])),
                                CONFIG=agent_config
                            )
                        # sim_agent = SimAgent(
                        #         name='agent-'+str(len(self.sim_agents)+1),
                        #         id_color=COLORS[len(self.sim_agents)],
                        #         location=(int(loc[0]), int(loc[1])))
                        self.sim_agents.append(sim_agent)
                
                elif phase == 4:
                    # import pdb; pdb.set_trace()                    
                    occupied = set()
                    for rep in line:
                        # check if the object is a tomato, lettuce, onion, or plate
                        if rep in "tlop":
                            while True:
                                obj_counter = random.choice(self.world.objects["Counter"])
                                if (obj_counter.location not in occupied):
                                    occupied.add(obj_counter.location)
                                    break

                            obj = Object(
                                        location=obj_counter.location,
                                        contents=RepToClass[rep]())
                            self.world.insert(obj=obj)
                            obj_counter.acquire(obj=obj)

        self.distances = {}
        self.world.width = x+1
        self.world.height = y
        self.world.perimeter = 2*(self.world.width + self.world.height)

    def reset(self):
        self.world = World(arglist=self.arglist)
        self.recipes = []
        self.sim_agents = []
        self.agent_actions = {}
        self.t = 0

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

        # Load world & distances.
        self.load_level(
                level=self.arglist.level,
                num_agents=self.arglist.num_agents)

        self.all_subtasks = self.run_recipes()
        self.goal_objects_count = len(self.all_subtasks) * [0]
        self.completed_subtasks = len(self.all_subtasks) * [0]

        self.world.make_loc_to_gridsquare()
        self.world.make_reachability_graph()
        self.cache_distances()

    def close(self):
        return

    def step(self, action_dict):
        # Track internal environment info.
        self.t += 1

        # Get actions.
        for sim_agent in self.sim_agents:
            sim_agent.action = action_dict[sim_agent.name]

        # Check collisions.
        self.check_collisions()
        # self.obs_tm1 = copy.copy(self)

        # Execute.
        self.execute_navigation()

        # Check if done.
        done = self.done()

        # Visualize.
        self.display()

        reward = self.reward()
        info = {"t": self.t,
                "repr_obs": self.rep,
                "done": done, 
                "termination_info": self.termination_info,
                "agent_0_reward_shaping": self.calculate_reward_shaping(self.sim_agents[0]),
                "agent_1_reward_shaping": self.calculate_reward_shaping(self.sim_agents[1])
                }
        
        return reward, done, info

    def done(self):
        # Done if the episode maxes out
        if self.t >= self.arglist.max_num_timesteps and self.arglist.max_num_timesteps:
            self.termination_info = "Terminating because passed {} timesteps".format(
                    self.arglist.max_num_timesteps)
            self.successful = False
            return True

        assert any([isinstance(subtask, recipe.Deliver) for subtask in self.all_subtasks]), "no delivery subtask"

        # Done if subtask is completed.
        for subtask in self.all_subtasks:
            # Double check all goal_objs are at Delivery.
            if isinstance(subtask, recipe.Deliver):
                _, goal_obj = nav_utils.get_subtask_obj(subtask)

                delivery_loc = list(filter(lambda o: o.name=='Delivery', self.world.get_object_list()))[0].location
                goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
                if not any([gol == delivery_loc for gol in goal_obj_locs]):
                    self.termination_info = ""
                    self.successful = False

                    return False

        self.termination_info = "Terminating because all deliveries were completed"
        self.successful = True

        return True

    def calculate_reward_shaping(self, agent):        
        """Returns distance reward for agent under subtask, giving a reward when agents get closer and penalty when they get farther away."""
        MAX_PATH = self.world.perimeter + 1

        total_penalty = 0

        # First, look at the CHOP subtasks.
        unchopped_object_distances = []
        agent_going_for_chop = False
        min_cutting_board_distance = 0
        for i, subtask in enumerate(self.all_subtasks):
            if isinstance(subtask, recipe.Chop):
                if self.completed_subtasks[i] == 0:
                    start_obj, goal_obj = nav_utils.get_subtask_obj(subtask=subtask)
                    subtask_action_obj = nav_utils.get_subtask_action_obj(subtask=subtask)

                    start_object_location = self.world.get_all_object_locs(obj=start_obj)[0]
                    distance = self.world.get_path_distance_between(agent.location, start_object_location)
                    unchopped_object_distances.append(
                        distance
                    )

                    if distance == 0:
                        cutting_board_distances = []
                        for subtask_action_obj_loc in self.world.get_all_object_locs(obj=subtask_action_obj):
                            distance = self.world.get_path_distance_between(agent.location, subtask_action_obj_loc)
                            cutting_board_distances.append(
                                distance
                            )
                        min_cutting_board_distance = min(cutting_board_distances)

        if len(unchopped_object_distances) > 0 and not agent_going_for_chop:
            total_penalty += ((min(unchopped_object_distances) + MAX_PATH) + (len(unchopped_object_distances) - 1) * 2 * MAX_PATH) / MAX_PATH
        
        if agent_going_for_chop:
            total_penalty += (min_cutting_board_distance + (len(unchopped_object_distances) - 1) * 2 * MAX_PATH) / MAX_PATH

        # TODO: ADD MERGE
        # recipe_object_locations = []
        # for i, subtask in enumerate(self.all_subtasks):
        #     if isinstance(subtask, recipe.Chop):
        #         _, goal_obj = nav_utils.get_subtask_obj(subtask=subtask)
        #         if self.completed_subtasks[i] == 1:
        #             recipe_object_locations.extend((goal_obj.name, x) for x in self.world.get_all_object_locs(obj=goal_obj))
        #         else:
        #             recipe_object_locations.append((goal_obj.name, None))

        recipe_items = []
        recipe_items.append(Plate().name)
        recipe_items.extend([r.name for r in self.recipes[0].contents])

        recipe_object_locations = {}
        for ingredient in recipe_items:
            recipe_object_locations[ingredient] = []

        objs = []
        for o in self.world.objects.values():
            objs += o
        for obj in objs:
            if isinstance(obj, Object):
                for content in obj.contents:
                    if content.name in recipe_items:
                        try: 
                            recipe_object_locations[content.name].append(obj.location)
                        except:
                            import pdb; pdb.set_trace()

        distance_pairs = []
        for item1, item2 in combinations(recipe_object_locations.keys(), 2):
            item1_locs = recipe_object_locations[item1]
            item2_locs = recipe_object_locations[item2]

            if len(item1_locs) > 0 and len(item2_locs) > 0:
                min_distance = MAX_PATH
                for loc_1 in item1_locs:
                    for loc_2 in item2_locs:
                        distance = self.world.get_path_distance_between(loc_1, loc_2)
                        min_distance = min(min_distance, distance)

                if (min_distance != 0):
                    distance_pairs.append(min_distance)
                # total_distance += min_distance
                # num_distances += 1
            else:
                distance_pairs.append(MAX_PATH)

        # import pdb; pdb.set_trace()
        if (len(distance_pairs) > 0):
            if (total_penalty == 0):
                total_penalty += (min(distance_pairs) + (len(distance_pairs) - 1) * MAX_PATH) / MAX_PATH
            else:
                total_penalty += (len(distance_pairs) * MAX_PATH) / MAX_PATH

        def manhattan_distance(point1, point2):
            return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

        # Next, look at the DELIVER subtasks.
        # NOTE: ONLY CAN HANDLE 1 DELIVERY SUBTASK, ALTER IN FUTURE
        for i, subtask in enumerate(self.all_subtasks):
            if isinstance(subtask, recipe.Deliver):
                if self.completed_subtasks[i] == 0:
                    start_obj, _ = nav_utils.get_subtask_obj(subtask=subtask)
                    start_object_location = self.world.get_all_object_locs(obj=start_obj)
                    subtask_action_obj = nav_utils.get_subtask_action_obj(subtask=subtask)

                    if len(start_object_location) == 0:
                        total_penalty += 2
                    else:
                        distance = self.world.get_path_distance_between(agent.location, start_object_location[0]) + manhattan_distance(agent.location, start_object_location[0])
                        if distance == 0:
                            delivery_distances = []
                            for subtask_action_obj_loc in self.world.get_all_object_locs(obj=subtask_action_obj):
                                distance = self.world.get_path_distance_between(agent.location, subtask_action_obj_loc) + manhattan_distance(agent.location, subtask_action_obj_loc)
                                delivery_distances.append(
                                    distance
                                )
                            min_delivery_distance = min(delivery_distances)
                            total_penalty += min_delivery_distance / MAX_PATH
                            # print("DELIVER SHAPING:", min_delivery_distance / MAX_PATH)
                            # print(str(self))
                        else:
                            total_penalty += distance / MAX_PATH + 1
                            # print("DELIVER SHAPING 2:", distance / MAX_PATH + 1)
                            # print(str(self))

        return total_penalty
    
    def subtask_reward(self, index, subtask):
        if isinstance(subtask, recipe.Deliver):
            _, goal_obj = nav_utils.get_subtask_obj(subtask)
            delivery_loc = list(filter(lambda o: o.name=='Delivery', self.world.get_object_list()))[0].location
            goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
            if any([gol == delivery_loc for gol in goal_obj_locs]):
                print("Subtask Deliver Completed: {}".format(subtask))
                return 3
        else:
            _, goal_obj = nav_utils.get_subtask_obj(subtask)
            if len(self.world.get_all_object_locs(obj=goal_obj)) > self.goal_objects_count[index]:
                print("Subtask Other Completed: {}".format(subtask))
                # import pdb; pdb.set_trace()
                self.goal_objects_count[index] = len(self.world.get_all_object_locs(obj=goal_obj))
                return 1
            
            self.goal_objects_count[index] = len(self.world.get_all_object_locs(obj=goal_obj))

        return 0

    def reward(self):
        reward = 0
        for i, subtask in enumerate(self.all_subtasks):
            subtasks_reward = self.subtask_reward(i, subtask)
            reward += subtasks_reward

            if (subtasks_reward != 0):
                self.completed_subtasks[i] = 1
                print("Completed: ", self.completed_subtasks)
                print(str(self))
        
        # print("==== new frame ====")
        
        return reward

    def print_agents(self):
        for sim_agent in self.sim_agents:
            sim_agent.print_status()

    def display(self):
        self.update_display()
        # print(str(self))

    def update_display(self):
        self.rep = self.world.update_display()
        for agent in self.sim_agents:
            x, y = agent.location
            self.rep[y][x] = str(agent)


    def get_agent_names(self):
        return [agent.name for agent in self.sim_agents]

    def run_recipes(self):
        """Returns different permutations of completing recipes."""
        self.sw = STRIPSWorld(world=self.world, recipes=self.recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions
        subtasks = self.sw.get_subtasks(max_path_length=self.arglist.max_num_subtasks)
        all_subtasks = [subtask for path in subtasks for subtask in path]
        # print('Subtasks:', all_subtasks, '\n')
        return all_subtasks

    def get_AB_locs_given_objs(self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Returns list of locations relevant for subtask's Merge operator.

        See Merge operator formalism in our paper, under Fig. 11:
        https://arxiv.org/pdf/2003.11778.pdf"""

        # For Merge operator on Chop subtasks, we look at objects that can be
        # chopped and the cutting board objects.
        if isinstance(subtask, recipe.Chop):
            # A: Object that can be chopped.
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False) + list(map(lambda a: a.location,\
                list(filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))

            # B: Cutboard objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

        # For Merge operator on Deliver subtasks, we look at objects that can be
        # delivered and the Delivery object.
        elif isinstance(subtask, recipe.Deliver):
            # B: Delivery objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

            # A: Object that can be delivered.
            A_locs = self.world.get_object_locs(
                    obj=start_obj, is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))
            A_locs = list(filter(lambda a: a not in B_locs, A_locs))

        # For Merge operator on Merge subtasks, we look at objects that can be
        # combined together. These objects are all ingredient objects (e.g. Tomato, Lettuce).
        elif isinstance(subtask, recipe.Merge):
            A_locs = self.world.get_object_locs(
                    obj=start_obj[0], is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj[0], self.sim_agents))))
            B_locs = self.world.get_object_locs(
                    obj=start_obj[1], is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj[1], self.sim_agents))))

        else:
            return [], []

        return A_locs, B_locs

    def get_lower_bound_for_subtask_given_objs(
            self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Return the lower bound distance (shortest path) under this subtask between objects."""
        assert len(subtask_agent_names) <= 2, 'passed in {} agents but can only do 1 or 2'.format(len(agents))

        # Calculate extra holding penalty if the object is irrelevant.
        holding_penalty = 0.0
        for agent in self.sim_agents:
            if agent.name in subtask_agent_names:
                # Check for whether the agent is holding something.
                if agent.holding is not None:
                    if isinstance(subtask, recipe.Merge):
                        continue
                    else:
                        if agent.holding != start_obj and agent.holding != goal_obj:
                            # Add one "distance"-unit cost
                            holding_penalty += 1.0
        # Account for two-agents where we DON'T want to overpenalize.
        holding_penalty = min(holding_penalty, 1)

        # Get current agent locations.
        agent_locs = [agent.location for agent in list(filter(lambda a: a.name in subtask_agent_names, self.sim_agents))]
        A_locs, B_locs = self.get_AB_locs_given_objs(
                subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                start_obj=start_obj,
                goal_obj=goal_obj,
                subtask_action_obj=subtask_action_obj)

        # Add together distance and holding_penalty.
        return self.world.get_lower_bound_between(
                subtask=subtask,
                agent_locs=tuple(agent_locs),
                A_locs=tuple(A_locs),
                B_locs=tuple(B_locs)) + holding_penalty

    def is_collision(self, agent1_loc, agent2_loc, agent1_action, agent2_action):
        """Returns whether agents are colliding.

        Collisions happens if agent collide amongst themselves or with world objects."""
        # Tracks whether agents can execute their action.
        execute = [True, True]

        # Collision between agents and world objects.
        agent1_next_loc = tuple(np.asarray(agent1_loc) + np.asarray(agent1_action))
        if self.world.get_gridsquare_at(location=agent1_next_loc) == None or self.world.get_gridsquare_at(location=agent1_next_loc).collidable:
            # Revert back because agent collided.
            agent1_next_loc = agent1_loc
            
        agent2_next_loc = tuple(np.asarray(agent2_loc) + np.asarray(agent2_action))
        if self.world.get_gridsquare_at(location=agent2_next_loc) == None or self.world.get_gridsquare_at(location=agent2_next_loc).collidable:
            # Revert back because agent collided.
            agent2_next_loc = agent2_loc

        # Inter-agent collision.
        if agent1_next_loc == agent2_next_loc:
            if agent1_next_loc == agent1_loc and agent1_action != (0, 0):
                execute[1] = False
            elif agent2_next_loc == agent2_loc and agent2_action != (0, 0):
                execute[0] = False
            else:
                execute[0] = False
                execute[1] = False

        # Prevent agents from swapping places.
        elif ((agent1_loc == agent2_next_loc) and
                (agent2_loc == agent1_next_loc)):
            execute[0] = False
            execute[1] = False
        return execute

    def check_collisions(self):
        """Checks for collisions and corrects agents' executable actions.

        Collisions can either happen amongst agents or between agents and world objects."""
        execute = [True for _ in self.sim_agents]

        # Check each pairwise collision between agents.
        for i, j in combinations(range(len(self.sim_agents)), 2):
            agent_i, agent_j = self.sim_agents[i], self.sim_agents[j]
            exec_ = self.is_collision(
                    agent1_loc=agent_i.location,
                    agent2_loc=agent_j.location,
                    agent1_action=agent_i.action,
                    agent2_action=agent_j.action)

            # Update exec array and set path to do nothing.
            if not exec_[0]:
                execute[i] = False
            if not exec_[1]:
                execute[j] = False

            # Track collisions.
            if not all(exec_):
                collision = CollisionRepr(
                        time=self.t,
                        agent_names=[agent_i.name, agent_j.name],
                        agent_locations=[agent_i.location, agent_j.location])
                self.collisions.append(collision)

        # print('\nexecute array is:', execute)

        # Update agents' actions if collision was detected.
        for i, agent in enumerate(self.sim_agents):
            if not execute[i]:
                agent.action = (0, 0)
            # print("{} has action {}".format(color(agent.name, agent.color), agent.action))

    def execute_navigation(self):
        for agent in self.sim_agents:
            interact(agent=agent, world=self.world)
            self.agent_actions[agent.name] = agent.action


    def cache_distances(self):
        """Saving distances between world objects."""
        counter_grid_names = [name for name in self.world.objects if "Supply" in name or "Counter" in name or "Delivery" in name or "Cut" in name]
        # Getting all source objects.
        source_objs = copy.copy(self.world.objects["Floor"])
        for name in counter_grid_names:
            source_objs += copy.copy(self.world.objects[name])
        # Getting all destination objects.
        dest_objs = source_objs

        # From every source (Counter and Floor objects),
        # calculate distance to other nodes.
        for source in source_objs:
            self.distances[source.location] = {}
            # Source to source distance is 0.
            self.distances[source.location][source.location] = 0
            for destination in dest_objs:
                # Possible edges to approach source and destination.
                source_edges = [(0, 0)] if not source.collidable else World.NAV_ACTIONS
                destination_edges = [(0, 0)] if not destination.collidable else World.NAV_ACTIONS
                # Maintain shortest distance.
                shortest_dist = np.inf
                for source_edge, dest_edge in product(source_edges, destination_edges):
                    try:
                        dist = nx.shortest_path_length(self.world.reachability_graph, (source.location,source_edge), (destination.location, dest_edge))
                        # Update shortest distance.
                        if dist < shortest_dist:
                            shortest_dist = dist
                    except:
                        continue
                # Cache distance floor -> counter.
                self.distances[source.location][destination.location] = shortest_dist

        # Save all distances under world as well.
        self.world.distances = self.distances

