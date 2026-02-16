#Creates the decision problem from the raw file

from modules.tree import Node, InfoSet
from modules.utils import make_rational

def construct_game_tree(path_to_file):

    root = Node(description='/', parent='root')
    nodes_dict = {'/': root}
    infosets_dict = {}
    infoset_keys = []
    id_node = 0
    id_infoset = 0


    with open(path_to_file, "r") as game:
        #Go line by line and add nodes with their info set info to the game tree and store the info sets. Also check that the info sets have indeed unique action sets assigned
        for line in game:
            #ignore commented lines
            if line[0] == "#":
                continue

            parts = line.split()
            
            if parts[0] == "node":
                node_descr = parts[1]
                try:
                    current = nodes_dict[node_descr]
                except KeyError:
                    raise Exception("The following node has never been constructed!:", node_descr)

                if hasattr(current, 'type'):
                    raise Exception("The current node seems to have been constructed fully before. For example, its type has already been defined.:", node_descr)


                current.add_id(id_node)
                id_node += 1

                #Extract the info to that node in this line
                if parts[2] == "chance":
                    current.type = "chance"
                    current.actions = []
                    current.action_probs = {}
                    for word in parts[4:]:
                        current.actions.append( word.split("=")[0] )
                        current.action_probs[ word.split("=")[0] ] = make_rational( word.split("=")[1] )
                    
                elif parts[2] == "player":
                    current.type = "player"
                    current.player = parts[3]
                    current.actions = parts[5:]

                elif parts[2] + " " + parts[3] == "leaf payoffs":
                    current.type = "leaf"
                    current.payoffs = []
                    for word in parts[4:]:
                        current.payoffs.append( make_rational( word.split("=")[1] ) )
                
                else:
                    raise Exception("Something is wrong with the game file, unexpected word in the third position of the following line: ", line)

                if current.type != "leaf":
                    current.add_children()
                    for act, child in current.children.items():
                        nodes_dict[child.descr] = child

            elif parts[0] == "infoset":
                infoset = InfoSet(parts[1], parts[3:])
                infoset.add_id(id_infoset)
                id_infoset += 1
                infosets_dict[parts[1]] = infoset
                infoset_keys.append(parts[1])

            else:
                raise Exception("Something is wrong with the game file, unexpected word in the first position of the following line: ", line)


        for descr, infoset in infosets_dict.items():
            actions = []
            player = None
            num_changes_actions = 0
            num_changes_player = 0

            for node_descr in infoset.node_descrs:
                try:
                    current = nodes_dict[node_descr]
                except KeyError:
                    raise Exception("The following node has never been constructed!:", node_descr)

                if not hasattr(current, 'type'):
                    raise Exception("First you have to update the type of this node before dealing with this infoset!", node_descr)
                
                current.add_inf(infoset)

                if actions != current.actions:
                    actions = current.actions
                    num_changes_actions += 1                

                if player != current.player:
                    player = current.player
                    num_changes_player += 1                

            if num_changes_actions != 1:
                raise Exception("The info set named as the following has an ill-defined action set: ", descr)
            if num_changes_player != 1:
                raise Exception("The info set named as the following has an ill-defined player assignment: ", descr)

            infoset.add_pl_acts(player, actions)

    return nodes_dict, infosets_dict, infoset_keys