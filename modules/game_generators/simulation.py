# A test in simulation game: there are n test scenarios in total, the simulator has up to m tests available, and the AI will be deployed in k scenarios.

import yaml
import numpy as np
import os
from modules.tree import Node, InfoSet
from modules.utils import afewletters
from modules.game_generators.writegamefile import write
from math import factorial
import random
from copy import deepcopy

script_dir = os.path.dirname(os.path.abspath(__file__))
library_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

class generategame():
    #Generates a simulation game according to parameters found in the file

    def __init__(self, specifications):
        if isinstance(specifications, dict):
            self.specifications = specifications
        else:
            with open(specifications, 'r') as file:
                self.specifications = yaml.safe_load(file)
    
    def generate(self, output_instr={'local':True, 'output_dir':'EFGs/simulation'}):
        if not hasattr(self, 'infosets_dict'):
            self.generate_game()
        
        if self.success:
            self.save(output_instr)
        
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
        root = Node(description='/', parent='root')
        self.nodes_dict = {}
        self.node_counter = 0
        self.infosets_dict = {}
        self.infoset_keys = []
        self.num_scenarios = self.specifications['scenarios']
        self.list_scenarios = [ str(i) for i in range(self.num_scenarios) ]
        if self.specifications['decision_node_branching']['method'] == 'exact' and self.specifications['decision_node_branching']['width'] == 2:
            self.player_actions = ['g', 'b']
        else:
            raise Exception("We haven't defined things for more than two actions (good and bad) yet!")
            # self.player_actions = pl_action_labels(self.specifications['decision_node_branching']['width'])

        assert self.specifications['payoffs']['reality_payoffs'] == 'uniform'
        g_bounds = self.specifications['payoffs']['good_in_real_bounds']
        b_bounds = self.specifications['payoffs']['bad_in_real_bounds']
        self.rewards = []
        for i in range(self.num_scenarios):
            infoset = InfoSet(f'inf{i}/', [])
            infoset.add_id(i)
            infoset.add_pl_acts('1', self.player_actions)
            self.infosets_dict[infoset.descr] = infoset
            self.infoset_keys.append(infoset.descr)

            self.rewards.append( { 'g': random.uniform( g_bounds[0] , g_bounds[1]), 'b': random.uniform( b_bounds[0] , b_bounds[1]) } )

        self.specifications['drawn_rewards'] = [str(i) + ' :  g -> ' + str(self.rewards[i]['g']) + ' , b -> ' + str(self.rewards[i]['b']) for i in range(self.num_scenarios)]

        self.num_tests_max = self.specifications['testing']['num_tests_max']
        assert self.specifications['testing']['method'] == 'chance'
        self.reroll_testing = self.specifications['testing']['sim_reroll_prob']
        self.num_deploy_max = self.specifications['reality']['num_deploy_max']
        assert self.num_deploy_max >= 1
        assert self.specifications['reality']['method'] == 'chance'
        self.reroll_reality = self.specifications['reality']['real_reroll_prob']

        
        #number of tests/deployments already done, what scenario number was drawn last time, list of how often each scenario has been drawn so far
        details = [0,0,[0] * self.num_scenarios]
        self.build_simulation(root, details)

        self.success = True

    def build_simulation(self, current, details):
        if details[0] >= self.num_tests_max:
            self.build_reality(current, multinom_coeff(details[2]) )
        else:
            self.add_simchance_level(current, details)

    def add_simchance_level(self, node, details):
        # node indicates which node we want to be a chance node, details indicate the details described above
        node.add_id(self.node_counter)
        self.node_counter += 1
        
        node.type = 'chance'
        nonrepetitive_scenarios = self.list_scenarios[details[1]:]
        # I still believe we have to divide by num_scenarios, so that, for example, draw (2,2,2) is just as likely as draw (0,0,0). As a counterbalance, we are multiplying the leaf payoffs by the multinomial coefficient!
        scenario_prob = self.reroll_testing / self.num_scenarios
        node.action_probs = {scn: scenario_prob for scn in nonrepetitive_scenarios}
        node.action_probs['e'] = 1 -  scenario_prob * len(nonrepetitive_scenarios)
        node.actions = nonrepetitive_scenarios + ['e']
        node.add_children()
        self.nodes_dict[node.descr] = node

        for scn in nonrepetitive_scenarios: 
            sim_decision = node.gotochild(scn)
            self.complete_sim_node(sim_decision, int(scn), details)
        
        deployment_root = node.gotochild('e')
        self.build_reality(deployment_root, multinom_coeff(details[2]) )


    def complete_sim_node(self, node, inf_id, details):
        node.add_id(self.node_counter)
        self.node_counter += 1
        
        node.type = 'player'
        node.player = '1'
        node.actions = self.player_actions
        infoset = self.infosets_dict[f'inf{inf_id}/']
        node.infoset = infoset
        infoset.add_node(node)
        node.add_children()
        self.nodes_dict[node.descr] = node

        leaf = node.gotochild('b')
        leaf.add_id(self.node_counter)
        self.node_counter += 1        
        leaf.type = 'leaf'
        # if self.node_counter >= 2* 1e6:
        #         raise Exception("Reached 20 million nodes. For safety, we will raise an error here for now.")
        
        leaf.payoffs = [ self.specifications['payoffs']['screened'] ]
        self.nodes_dict[leaf.descr] = leaf

        new_details = deepcopy(details)
        new_details[0] += 1
        new_details[1] = inf_id
        new_details[2][inf_id] += 1
        self.build_simulation(node.gotochild('g'), new_details)


    # util_multiplier is the integer of in how many different orders the simulated scenarios could have been drawn. It is to be multiplied with the payoffs in the leaf nodes!
    def build_reality(self, node, util_multiplier):
        node.add_id(self.node_counter)
        self.node_counter += 1
        
        node.type = 'chance'
        first_scenario_prob = 1 / self.num_scenarios
        node.action_probs = {scn: first_scenario_prob for scn in self.list_scenarios}
        node.actions = self.list_scenarios
        node.add_children()
        self.nodes_dict[node.descr] = node

        for infoset_id, scn in enumerate(self.list_scenarios): 
            real_decision = node.gotochild(scn)
            #list of [deployment scenario id, action] so far, list of how often each scenario has been drawn so far
            details = [[],[0] * self.num_scenarios]
            self.complete_real_decision(real_decision, infoset_id, details, util_multiplier)        
        
    def complete_real_decision(self, node, inf_id, details, util_multiplier):
        node.add_id(self.node_counter)
        self.node_counter += 1
        
        node.type = 'player'
        node.player = '1'
        node.actions = self.player_actions
        infoset = self.infosets_dict[f'inf{inf_id}/']
        node.infoset = infoset
        infoset.add_node(node)
        node.add_children()
        self.nodes_dict[node.descr] = node

        bchild = node.gotochild('b')
        b_details = deepcopy(details)
        b_details[0].append([inf_id,'b'])
        b_details[1][inf_id] += 1 
        if len(b_details[0]) >= self.num_deploy_max:
            self.make_real_leaf(bchild, b_details[0], util_multiplier * multinom_coeff(b_details[1]) )
        else:
            self.complete_real_chance(bchild, b_details, util_multiplier)

        gchild = node.gotochild('g')
        g_details = deepcopy(details)
        g_details[0].append([inf_id,'g'])
        g_details[1][inf_id] += 1 
        if len(g_details[0]) >= self.num_deploy_max:
            self.make_real_leaf(gchild, g_details[0], util_multiplier * multinom_coeff(g_details[1]) )
        else:
            self.complete_real_chance(gchild, g_details, util_multiplier)
        
    def complete_real_chance(self, node, details, util_multiplier):
        node.add_id(self.node_counter)
        self.node_counter += 1
        
        node.type = 'chance'
        # details[0][-1][0] is the first entry of the last [deployment scenario id, action] in the history of actions, so practically, the last infoset id seen
        nonrepetitive_scenarios = self.list_scenarios[details[0][-1][0]:]
        # As argued previously, I believe we have to divide by num_scenarios still
        scenario_prob = self.reroll_reality / self.num_scenarios
        node.action_probs = {scn: scenario_prob for scn in nonrepetitive_scenarios}
        node.action_probs['e'] = 1 -  scenario_prob * len(nonrepetitive_scenarios)
        node.actions = nonrepetitive_scenarios + ['e']
        node.add_children()
        self.nodes_dict[node.descr] = node

        for scn in nonrepetitive_scenarios:
            self.complete_real_decision(node.gotochild(scn), int(scn), details, util_multiplier)         
        
        self.make_real_leaf(node.gotochild('e'), details[0], util_multiplier * multinom_coeff(details[1]) )

    def make_real_leaf(self, node, history, util_multiplier):       
        node.add_id(self.node_counter)
        self.node_counter += 1        
        node.type = 'leaf'
        # if self.node_counter >= 2* 1e6:
        #         raise Exception("Reached 20 million nodes. For safety, we will raise an error here for now.")
        
        pay = 0
        for scn, act in history:
            pay += self.rewards[scn][act]
        node.payoffs = [ pay * util_multiplier]
        self.nodes_dict[node.descr] = node


# Given a list that indicates how many times a particular scenario is drawn, how many orders (permutations) are there of drawing scenarios to these particular counts?: 
# https://en.wikipedia.org/wiki/Multinomial_theorem#Number_of_unique_permutations_of_words
def multinom_coeff(list): 
    #integer list of length n, where there are n scenarios or items to pick from overall. sum(list) is the number of simulations/deployments/draws done overall
    numerator = factorial(sum(list))
    denominator = np.prod([factorial(x) for x in list])
    return round(numerator / denominator)


def pl_action_labels(n):
    if n == 2:
        return ['g', 'b']
    elif n == 3:
        return ['g', 'p', 'b']
    else:
        return afewletters(n) 
