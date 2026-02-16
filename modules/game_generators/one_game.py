import argparse
import os
import random
import yaml
import numpy as np

from modules.utils import change_yaml
from modules.game_generators.detection import generategame as detection_generate
from modules.game_generators.random import generategame as random_generate
from modules.game_generators.simulation import generategame as simulation_generate


sizes = {'a':1e3, 'b':1e4, 'c':1e5, 'xs': 1e6, 's': 1e7, 'm': 2e7, 'l': 4e7, 'xl': 1e8}

detection_numbers = {
    'a': {'num_graphnodes':10, 'num_rounds':2}, 
    'b': {'num_graphnodes':10, 'num_rounds':3}, 
    'c': {'num_graphnodes':13, 'num_rounds':3}, 
    'xs': {'num_graphnodes':10, 'num_rounds':4}, 
    's': {'num_graphnodes':13, 'num_rounds':4}, 
    'm': {'num_graphnodes':11, 'num_rounds':5}, 
    'l': {'num_graphnodes':12, 'num_rounds':5}, 
    'xl': {'num_graphnodes':11, 'num_rounds':6}
    }
graph_types = ['gnp', 'gnm', 'grid']

random_numbers = {
    'a': {'depth':[4,8]},
    'b': {'depth':[5,9]},
    'c': {'depth':[6,10]},
    'xs': {'depth':[7,12]}, 
    's': {'depth':[8,13]}, 
    'm': {'depth':[9,14]}, 
    'l': {'depth':[10,14]}, 
    'xl': {'depth':[10,15]}
    }
def compute_factor(i,max):
    return 0.5 + ( (i-1) / (max-1) )

simulation_numbers = {
    'a': {'scenarios':1, 'num_tests_max':6}, 
    'b': {'scenarios':2, 'num_tests_max':6}, 
    'c': {'scenarios':2, 'num_tests_max':8}, 
    'xs': {'scenarios':3, 'num_tests_max':7}, 
    's': {'scenarios':3, 'num_tests_max':8}, 
    'm': {'scenarios':3, 'num_tests_max':9}, 
    'l': {'scenarios':3, 'num_tests_max':10}, 
    'xl': {'scenarios':3, 'num_tests_max':11}
    }



def main():
    parser = argparse.ArgumentParser(description='Generate game with parameters')
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--size', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--i', type=int, required=False, default=1)
    parser.add_argument('--num_per_size', type=int, required=False, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.type == "random":
        generate_random(args)
    elif args.type == "detection":
        generate_detection(args)
    elif args.type == "simulation":
        generate_simulation(args)
    elif args.type == "team":
        generate_team(args)
    else:
        raise ValueError(f"Unknown game type: {args.type}")
         
def generate_random(args):
    with open("./decisionproblems/generation_instructions/randomstandard.yaml", "r") as file:
        standard_specs = yaml.safe_load(file)

    depth = random_numbers[args.size]['depth']
    pl_degrees = [3,5]
    ch_degrees = [3,5]
    factor = compute_factor(args.i, args.num_per_size)

    changes = {'tree_nodes': {'depth':depth, 'method':'bounds_linear', 'chance_presence': 0.2}, 'chance': {'width':ch_degrees, 'method': 'bounds_uniform', 'probs': 'uniform'}, 'decision_node_branching': {'width':pl_degrees, 'method': 'bounds_uniform'}, 'infosets': {'method': 'pm20', 'specs': 'nodesperinfoset', 'proportion': 'root', 'root_power': 0.66, 'factor': factor}}

    specs = change_yaml(standard_specs, changes)

    succeeded = False
    print(f"Generating game with depth {depth} and infosets of size {factor} * (total_num_nodes ** 0.66).")
    repeat = 1
    while not succeeded:
        game = random_generate(specs)
        succeeded = game.generate(output_instr={'local':False, 'output_dir':args.output_dir})
        if succeeded:
            print(f"Generation attempt {repeat} succeeded.")
        else:
            print(f"Generation attempt {repeat} failed. Will try again")
            repeat += 1
            if repeat > 20:
                raise ValueError("Game generation failed after 20 attempts. Will stop here.")
    

def generate_detection(args):
    with open("./decisionproblems/generation_instructions/detectionstandard.yaml", "r") as file:
        standard_specs = yaml.safe_load(file)

    num_graphnodes = detection_numbers[args.size]['num_graphnodes']
    num_rounds = detection_numbers[args.size]['num_rounds']
    num_samples = round( sizes[args.size] / ( (num_graphnodes-1) ** (num_rounds + 1) ) )
    
    type = graph_types[args.i % len(graph_types)]
    succeeded = False

    print(f"Generating game with a graph of type {type}with {num_graphnodes} nodes, {num_rounds} rounds of picking, and {num_samples} samples for pattern combinations.")
    repeat = 1
    while not succeeded:
        delete = []
        graph = {"distribution": type}
        if graph["distribution"] == "grid":
            m = round( np.sqrt(num_graphnodes) )
            n = round( num_graphnodes / m)
            graph["parameter_list"] = [m,n]
            num_graphnodes = n*m 
        elif graph["distribution"] == "gnp":
            p = random.uniform(0.2, 0.4)
            graph["parameter_list"] = [num_graphnodes,p]
        elif graph["distribution"] == "gnm":
            num_edges = num_graphnodes * random.randint(3,5)
            graph["parameter_list"] = [num_graphnodes, num_edges]
        else:
            raise ValueError("The distribution is not recognized. Please use 'grid' or 'gnp' or 'gnm'.")
        
        pattern_size_bounds  = [2,4]
        num_communities_bounds = [2,2]
        valuation_bounds = [1,10]

        communities = {}
        communities["method"] = "random"
        delete.append(["communities", "community_list"])
        delete.append(["communities", "valuation_list"])
        
        communities["num_samples"] = num_samples
        communities["num_communities_bounds"] = num_communities_bounds
        communities["community_size_bounds"] = pattern_size_bounds
        communities["valuation_bounds"] = valuation_bounds

        changes = {'graph': graph, 'communities': communities, 'num_rounds_max': num_rounds, 'delete': delete}

        specs = change_yaml(standard_specs, changes)
        game = detection_generate(specs)
        succeeded = game.generate(output_instr={'local':False, 'output_dir':args.output_dir})
        if succeeded:
            print(f"Generation attempt {repeat} succeeded.")
        else:
            print(f"Generation attempt {repeat} failed. Will try again")
            repeat += 1
            if repeat > 20:
                raise ValueError("Game generation failed after 20 attempts. Will stop here.")


def generate_simulation(args):

    with open("./decisionproblems/generation_instructions/simulationstandard.yaml", "r") as file:
        standard_specs = yaml.safe_load(file)

    scenarios = simulation_numbers[args.size]['scenarios']
    center = simulation_numbers[args.size]['num_tests_max']
    shift = args.i - ( args.num_per_size + 1)/2
    num_tests_max = center + shift
    num_deploy_max = round( 0.7*center - shift )

    changes = {'scenarios': scenarios, 'testing': {'num_tests_max':num_tests_max}, 'reality': {'num_deploy_max':num_deploy_max}}

    specs = change_yaml(standard_specs, changes)

    succeeded = False
    print(f"Generating game with {scenarios} scenarios, {num_tests_max} maximum number of simulations, and {num_deploy_max} maximum number of deployments.")
    repeat = 1
    while not succeeded:
        game = simulation_generate(specs)
        succeeded = game.generate(output_instr={'local':False, 'output_dir':args.output_dir})
        if succeeded:
            print(f"Generation attempt {repeat} succeeded.")
        else:
            print(f"Generation attempt {repeat} failed. Will try again")
            repeat += 1
            if repeat > 20:
                raise ValueError("Game generation failed after 20 attempts. Will stop here.")
    
if __name__ == "__main__":
    main()