# Generate a random game parametrized by info on the range or number of nodes, infosets, actions per infoset, degree of absentmindedness, chance nodes, and utility values.

import yaml
import random
import numpy as np
import functools
import os
import time

from modules.tree import Node, create_infoset
from modules.utils import afewletters, sample_from_simplex
from modules.game_generators.writegamefile import write

script_dir = os.path.dirname(os.path.abspath(__file__))
library_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

class generategame():
    #Generates a random game according to parameters found in the file

    def __init__(self, specifications):
        if isinstance(specifications, dict):
            self.specifications = specifications
        else:
            with open(specifications, 'r') as file:
                self.specifications = yaml.safe_load(file)
    
    def generate(self, output_instr={'local':True, 'output_dir':'EFGs/random'}):
        if not hasattr(self, 'infosets_dict'):
            self.generate_parametrized_game()
        
        if self.success:
            self.save(output_instr)
        
        return self.success


    def save(self, output_instr):
        if not hasattr(self, 'infosets_dict'):
            self.generate_parametrized_game()
        
        if self.success:

            if output_instr['local']:
                folder_dir = os.path.join(library_dir, output_instr['output_dir'])
            else:
                folder_dir = output_instr['output_dir']
            
            self.saved_dir, self.descr = write(folder_dir, self.specifications, self.nodes_dict, self.infosets_dict, self.infoset_keys)       

    def generate_parametrized_game(self):
        self.start_time = time.time()
        self.report_counter = 0
        self.generate_tree()       

        if self.success: 
            self.generate_infosets()    #Generates the infosets
    

    def generate_tree(self):
        root = Node(description='/', parent='root')
        self.nodes_dict = {}
        self.nodes_wo_inf = {}
        stack = [root]
        self.node_counter  = 0
        while len(stack) >= 1:
            current = stack.pop()
            self.nodes_dict[current.descr] = current
            current.add_id(self.node_counter)
            self.node_counter += 1
            if self.node_counter >= self.report_counter * 4e6:
                self.report_counter += 1
                print(f"Time passed : {time.time()-self.start_time}  ,  Node count: {self.node_counter}")
                if self.node_counter >= 2*1e8:
                        raise Exception("Reached 200 million nodes. For safety, we will raise an error here for now.")

            specs = self.sample_specs(current)
            current.update_type(specs)
            if current.type != 'leaf':
                current.add_children()
                children = list(current.children.values())
                stack.extend(reversed(children))
            

        if len(self.nodes_wo_inf) >= 1:
            self.success = True
        else:
            print("Created a game without any decision nodes. Will throw it away")
            self.success = False
    
    def sample_specs(self,node):
        #Determines what type the node should be, and how many outgoing edges it should have
        args = {}
        type = sample_type(node,self.specifications['tree_nodes'])
        args['type'] = type
        if type == 'player':
            args['player'] = sample_player(node) 
            actions = sample_num_player_actions(self.specifications['decision_node_branching'])
            args['actions'] = actions
            self.nodes_wo_inf.setdefault( len(actions) , {} )[node.descr] = node    #if self.nodes_wo_inf already has something under key len(actions), then use the dictionary that is there already, otherwise, we create an empty dictionary there and add a new value under key node.descr. This keeps track of all player nodes with outdegree len(actions)
        elif type == 'chance':
            args['actions'], args['action_probs'] = sample_chance_outcomes(self.specifications['chance'])
        elif type == 'leaf':
            args['payoffs'] = sample_payoff(self.specifications['payoffs'])
        else:
            raise Exception("Something went wrong with args['type'], it is invalid", args['type'])
        
        return args


    def generate_infosets(self):
        print(f"Time passed : {time.time()-self.start_time}  ,  Node count: {self.node_counter}  ,  Starting to generate infosets now.")
        self.infosets_dict = {}
        self.infoset_keys = []

        num_nodes_wo_infosets = 0
        for degree, incompl_nodes in self.nodes_wo_inf.items():
            num_nodes_wo_infosets += len(incompl_nodes)

        num_nodes_generator = InfosetCreator( self.specifications['infosets'], num_nodes_wo_infosets)

        current_infoset = 0
        for degree, incompl_nodes in self.nodes_wo_inf.items():
            while len(incompl_nodes) >= 1:
                num = next(num_nodes_generator)
                if num >= len(incompl_nodes):
                    selected_nodes = list(incompl_nodes)
                    num = len(incompl_nodes)
                else:
                    selected_nodes = random.sample( list(incompl_nodes), num)
                
                infoset_nodes = {}
                for i in range(num):
                    infoset_nodes[selected_nodes[i]] = incompl_nodes[ selected_nodes[i] ]
                    del incompl_nodes[ selected_nodes[i] ]
                
                name = 'inf' + str(current_infoset) + '/'

                if infoset_nodes == {}:
                    raise Exception("Something went wrong with the infoset generation, it is empty", infoset_nodes)
                self.infosets_dict[name] = create_infoset(infoset_nodes, name, current_infoset)
                self.infoset_keys.append(name)
                
                current_infoset += 1
            
        print(f"Time passed : {time.time()-self.start_time}  ,  Finished generating infosets.")

def sample_type(node, tree_specs):
    if tree_specs['method'] == 'bounds_linear':
        lower, upper = tree_specs['depth']

        if node.depth >= upper:
            return 'leaf'
        #sample if leaf
        elif node.depth >= lower:
            if random.random() < ( node.depth + 1 - lower ) / ( upper + 1 - lower ):
                return 'leaf'
        
        #sample if chance
        if random.random() < tree_specs['chance_presence']:
            return 'chance'
        else:
            return 'player'

def sample_player(node):      #To be changed if multiplayer!
    return '1'

def sample_num_player_actions(branching_specs):
    if branching_specs['method'] == 'bounds_uniform':
        width = random.randint(*branching_specs['width'])
    
    ### ADD MORE WIDTH GENERATIONS HERE ###
    
    return afewletters(width)

def sample_chance_outcomes(branching_specs):
    if branching_specs['method'] == 'bounds_uniform':
        width = random.randint(*branching_specs['width'])
    
    ### ADD MORE WIDTH GENERATIONS HERE ###
    
    if branching_specs['probs'] == 'uniform':
        probs = sample_from_simplex(width)

    return afewletters(width), probs

def sample_payoff(payoff_specs):
    if payoff_specs == 'uniform':
        return [ random.random() ]    #To be changed if multiplayer!
    
    ### ADD MORE WIDTH GENERATIONS HERE ###
    
    else:
        raise Exception("Payoff sampling not well-defined", payoff_specs)


def output_number(n):
    return n

def output_random_scale(z, max_ratio):
    return round ( ( 1 + random.uniform(-max_ratio, max_ratio) ) * z )
class InfosetCreator:
    def __init__(self, infoset_specs, total_num_nodes):
        assert infoset_specs['specs'] == 'nodesperinfoset'
        
        if infoset_specs['proportion'] == 'log':
            float = max( 1 , infoset_specs['factor'] * np.log2(total_num_nodes ) )
        elif infoset_specs['proportion'] == 'root':
            float = max( 1 , infoset_specs['factor'] * ( total_num_nodes ** infoset_specs['root_power'] ) )
        else:
            raise Exception("Proportion not well-defined", infoset_specs['proportion'])

        if infoset_specs['method'] == 'exact':
            self.outputter = functools.partial(output_number, round( float ) )
        elif infoset_specs['method'] == 'pm20':
            self.outputter = functools.partial(output_random_scale, float, 0.2)
        else:
            raise Exception("Method not well-defined", infoset_specs['method'])

    def __iter__(self):
        return self

    def __next__(self):
        return self.outputter()