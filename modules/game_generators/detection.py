# Games of detecting communities in graphs: there is a graph and some nodes of it are to be found (_positive_). They live in connected neighborhoods. When inquiring a node, we will only remember it if it is a positive one. We only have finitely many tries to find as many nodes as possible.

import yaml
import os
from copy import deepcopy
import ast
import time

from modules.tree import Node, InfoSet
from modules.game_generators.graph_generators import create_graph_and_patterncombis
from modules.game_generators.writegamefile import write


script_dir = os.path.dirname(os.path.abspath(__file__))
library_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

class generategame():
    #Generates a community detection game according to parameters found in the file

    def __init__(self, specifications):
        if isinstance(specifications, dict):
            self.specifications = specifications
        else:
            with open(specifications, 'r') as file: 
                self.specifications = yaml.safe_load(file)
    
    def generate(self, output_instr={'local':True, 'output_dir':'EFGs/detection'}):
        if not hasattr(self, 'infosets_dict'):
            self.generate_game()
        
        if self.success:
            self.save(output_instr)
        else:
            print("There is no constellation of the list of the desired patterns that fits in the generated graph. We will need to skip this game.")
        
        return self.success

    def save(self, output_instr):
        if not hasattr(self, 'infosets_dict'):
            self.generate_game()
        
        if output_instr['local']:
            folder_dir = os.path.join(library_dir, output_instr['output_dir'])
        else:
            folder_dir = output_instr['output_dir']
        
        self.saved_dir, self.descr = write(folder_dir, self.specifications, self.nodes_dict, self.infosets_dict, self.infoset_keys)       

    def generate_game(self):
        self.start_time = time.time()
        self.report_counter = 0

        self.success = True

        self.nodes_dict = {}
        self.node_counter = 0
        self.infosets_dict = {}
        self.infoset_keys = []
        self.inf_counter = 0

        graph, pattern_combis_info = create_graph_and_patterncombis(self.specifications)
        if self.specifications['communities']['method'] == 'enumerate':
            all_pattern_combis = pattern_combis_info
            constant_valuations = self.specifications['communities']['valuation_list']
        elif self.specifications['communities']['method'] == 'random':
            all_pattern_combis, all_valuations = pattern_combis_info
        else:
            raise ValueError("The method of community detection is not recognized. Please use 'enumerate' or 'random'.")
        
        num_subgames = len(all_pattern_combis)
        if num_subgames == 0:
            self.success = False
            return

        self.specifications['the_generated_graph'] = { 'nodes': str(list(graph.nodes())), 'edges': str(list(graph.edges())) }
        self.specifications['graph']['parameter_list'] = str(self.specifications['graph']['parameter_list'])
        for key in ["community_list", "valuation_list", "num_communities_bounds", "community_size_bounds", "valuation_bounds"]:
            if key in self.specifications['communities']:
                self.specifications['communities'][key] = str(self.specifications['communities'][key])

        self.pickable_actions = [ make_string(n) for n in graph.nodes() ]
        self.num_rounds = self.specifications['num_rounds_max']
        assert self.num_rounds >= 1, "The number of rounds has to be positive!"

        node = Node(description='/', parent='root')
        node.add_id(self.node_counter)
        self.node_counter += 1
        node.type = 'chance'
        action_prob = 1 / len(all_pattern_combis)
        action_strings  = [ patterns_to_str(pattern_combi) for pattern_combi in all_pattern_combis ]
        node.actions = action_strings
        node.action_probs = { act: action_prob for act in action_strings }
        try:
            node.add_children()
        except ValueError as e:
            self.success = False
            return
        
        self.nodes_dict[node.descr] = node        

        for i, (pattern_combi, string) in enumerate(zip(all_pattern_combis, action_strings)):
            
            print(f"Building subgame {i+1}/{num_subgames} now. Total time passed so far: {time.time()-self.start_time}.  Node count: {self.node_counter}.")

            first_decision = node.gotochild(string)
            if self.specifications['communities']['method'] == 'enumerate':
                self.complete_subgame(first_decision, pattern_combi, constant_valuations)
            elif self.specifications['communities']['method'] == 'random':
                self.complete_subgame(first_decision, pattern_combi, all_valuations[i])
            else:
                raise ValueError("The method of community detection is not recognized. Please use 'enumerate' or 'random'.")   
             
    def complete_subgame(self, node, pattern_combi, valuations):
        obs = set()
        node.cum_value = 0
        remaining_positives = deepcopy(pattern_combi) #[node.copy() for pattern in pattern_combi for node in pattern]
        current_attempt = 0
        
        node.add_id(self.node_counter)
        self.node_counter += 1
        node.type = 'player'
        node.player = '1'
        node.actions = [x for x in self.pickable_actions if x not in obs]
        
        obs_str = observation_to_infstr(obs)
        if obs_str not in self.infosets_dict:
            infoset = InfoSet(obs_str, [])
            infoset.add_id(self.inf_counter)
            self.inf_counter += 1
            infoset.add_pl_acts('1', node.actions)
            self.infosets_dict[infoset.descr] = infoset
            self.infoset_keys.append(infoset.descr)
        else:
            infoset = self.infosets_dict[obs_str]
        
        node.infoset = infoset
        infoset.add_node(node)
        node.add_children()
        self.nodes_dict[node.descr] = node  

        for act in node.actions:
            child = node.gotochild(act)
            self.build(act, child, node.cum_value, remaining_positives, obs, current_attempt+1, valuations)

    def build(self, action, node, current_value, rem_pos, obs, current_attempt, vals):
        observation = obs.copy()
        remaining_positives, index = find_index_and_remove( rem_pos , action)
        if index is None:
            node.cum_value = current_value
        else:
            node.cum_value = current_value + vals[index]
            observation.add(action)

        if current_attempt >= self.num_rounds or isempty(remaining_positives):
            node.add_id(self.node_counter)
            self.node_counter += 1        
            node.type = 'leaf'
            if self.node_counter >= self.report_counter * 4e6:
                self.report_counter += 1
                print(f"Time passed : {time.time()-self.start_time}  ,  Node count: {self.node_counter}")
                if self.node_counter >= 2*1e8:
                        raise Exception("Reached 200 million nodes. For safety, we will raise an error here for now.")
            
            node.payoffs = [node.cum_value]
            self.nodes_dict[node.descr] = node
        else:
            node.add_id(self.node_counter)
            self.node_counter += 1
            node.type = 'player'
            node.player = '1'
            node.actions = [x for x in self.pickable_actions if x not in observation]
            
            observation_str = observation_to_infstr(observation)
            if observation_str not in self.infosets_dict:
                infoset = InfoSet(observation_str, [])
                infoset.add_id(self.inf_counter)
                self.inf_counter += 1
                infoset.add_pl_acts('1', node.actions)
                self.infosets_dict[infoset.descr] = infoset
                self.infoset_keys.append(infoset.descr)
            else:
                infoset = self.infosets_dict[observation_str]
            
            node.infoset = infoset
            infoset.add_node(node)
            node.add_children()
            self.nodes_dict[node.descr] = node  

            for act in node.actions:
                child = node.gotochild(act)
                self.build(act, child, node.cum_value, remaining_positives, observation, current_attempt+1, vals)            


def patterns_to_str(patterns): 
    string = ""
    for i, pattern in enumerate(patterns):
        if i == 0:
            string += f"pat{i}-"
        else:   
            string += f"__pat{i}-"
        # nodes, _ = pattern
        for j, node in enumerate(pattern):
            if j == 0:
                string += f"{make_string(node)}"
            else:
                string += f"-{make_string(node)}"
    return string

def observation_to_infstr(observ): 
    string = "inf__"
    if len(observ) == 0:
        return string + "empty"
    else:
        for i, node in enumerate(observ):
            if i == 0:
                string += f"{make_string(node)}"
            else:   
                string += f"*{make_string(node)}"
    return string

def make_string(node):
    if isinstance(node, tuple) or isinstance(node, list):
        return '(' + ','.join(str(item) for item in node) + ')'
    elif isinstance(node, str):
        return node
    elif isinstance(node, int):
        return str(node)
    else:
        raise TypeError(f"Unsupported type: {type(node)}. Expected str, tuple, list, or int.")

def find_index_and_remove(remaining_positives, action):
    action_tuple = ast.literal_eval(action)
    rem_pos = deepcopy(remaining_positives)
    for i, rem_pattern_nodes in enumerate(rem_pos):
        if action_tuple in rem_pattern_nodes:
            rem_pattern_nodes.remove(action_tuple)
            return rem_pos, i
    return rem_pos, None  # node action not found, so return an invalid index

def isempty(list):
    for sublist in list:
        if len(sublist) > 0:
            return False
    return True