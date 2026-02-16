# Check if the format of your decision problem file is correct.

import numpy as np
from collections import Counter
from modules.utils import is_float_or_rational, make_rational

def check_format(path_to_file):
    
    player_nodes_list = []
    yet_to_be_visited_nodes = {"/"}
    infosets_list = []
    nodes_of_infosets_unordered = [] 


    with open(path_to_file, "r") as game:
        for line in game:
            if line[0] == "#":
                continue
            elif line.split()[0] == "node":
                parts = line.split()
                node_descr = parts[1]
                if node_descr in yet_to_be_visited_nodes:
                    yet_to_be_visited_nodes.remove(node_descr)
                else:
                    raise Exception("""The node description in this line does not describe a valid path described in the previous lines: """, line) 

                if parts[2] == "chance":
                    if parts[3] != "actions":
                        raise Exception("""The fourth word string in this chance node line should simply be "actions": """, line) 
                    if len(parts) <= 4:
                        raise Exception("""We are missing chance node actions in the following line: """, line) 
                    sum = 0
                    chance_action_names = []
                    for k in range(4, len(parts)):
                        subparts = parts[k].split("=")
                        if len(subparts) >= 3:
                            raise Exception("The following action description of the following chance node has too many subparts: ", parts[k], line)
                        chance_action_names.append(subparts[0])
                        if '/' in subparts[0] or ':' in subparts[0]:
                            raise Exception("Don't use / or : in the action description of the following chance node: ", parts[k], line)
                        
                        if node_descr == '/':  #Special case for attaching strings to this
                            yet_to_be_visited_nodes.add('/C:' + subparts[0])
                        else:
                            yet_to_be_visited_nodes.add(node_descr + '/C:' + subparts[0])
                    
                        if not is_float_or_rational(subparts[1]):
                            raise Exception("The following action description of the following chance node line does not include a float number!: ", parts[k], line)
                        sum += make_rational(subparts[1])
                    if len(set(chance_action_names)) != len(chance_action_names):
                        raise Exception("""We have duplicate chance action names in the following line: """, line)
                    if not np.isclose(sum, 1.0, atol=1e-16):
                        raise Exception("The following chance node has action probabilities that don't sum up to exactly 1: ", line)

                elif parts[2] == "player":
                    player_nodes_list.append(node_descr)
                    if int(parts[3]) != 1:
                        raise Exception("The following player node line is not assigned to player 1!: ", line)

                    if parts[4] != "actions":
                        raise Exception("""The fifth word string in this player node line should simply be "actions": """, line) 

                    if len(parts) <= 5:
                        raise Exception("""We are missing player node actions in the following line: """, line)
                    
                    if len(set(parts[5:])) != len(parts[5:]):
                        raise Exception("""We have duplicate player action names in the following line: """, line)
                                        
                    for k in range(5, len(parts)):
                        if '/' in parts[k] or ':' in parts[k]:
                            raise Exception("Don't use / or : in the action description of the following player node: ", parts[k], line)
                        
                        if node_descr == '/':  #Special case for attaching strings to this
                            yet_to_be_visited_nodes.add('/P' + parts[3] + ':' + parts[k])
                        else:
                            yet_to_be_visited_nodes.add(node_descr + '/P' + parts[3] + ':' + parts[k])
                
                elif parts[2] == "leaf":
                    if parts[3] != "payoffs":
                        raise Exception("""The third word string in this terminal node line should simply be "payoffs": """, line) 

                    if len(parts) <= 4:
                        raise Exception("""We are missing player payoffs in the following terminal node line: """, line)
                    

                    for k in range(4, len(parts)):
                        player_id, player_payoff = parts[k].split("=")
                       
                        if int(player_id) != 1:
                            raise Exception("The following payoff in the following terminal node line is not assigned to P1: ", parts[k], line)

                        if not is_float_or_rational(player_payoff):   
                            raise Exception("The following player payoff in the following terminal node must have a float number in the end!: ", parts[k], line)

                else:
                    raise Exception("The following line has a violating third word string for being a node line: ", line)


            elif line.split()[0] == "infoset":
                parts = line.split()
                infosets_list.append(parts[1])
                if parts[2] != "nodes":
                    raise Exception("""The third word string in this infoset line should simply be "nodes": """, line) 
            
                if len(parts) <= 3:
                    raise Exception("""We are missing nodes in the following infoset line: """, line)
                
                for k in range(3, len(parts)):
                    nodes_of_infosets_unordered.append(parts[k])
            else:
                raise Exception("The following line has a violating first word string: ", line)

    if len(player_nodes_list) == 0:
        raise Exception("There are no player node lines in this file.")
    
    if len(infosets_list) == 0:
        raise Exception("There are no infoset lines in this file. This should only be acceptable if the game shall be perfect recall and if defining extra infosets for each node is considered cumbersome for whatever reasons. I cannot guarantee that the solvers will work for this file though.")
    
    counter_nodes_lines = Counter(player_nodes_list)
    counter_infosets = Counter(nodes_of_infosets_unordered)
    if not counter_nodes_lines == counter_infosets:
        raise Exception("The node descriptions from the infoset lines do not fully match the node descriptions from the node lines!")
 
    return True

