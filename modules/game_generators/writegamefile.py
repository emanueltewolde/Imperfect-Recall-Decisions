import os
import re
import yaml
import time

def next_file_number(folder_path,folder_name):
    files = os.listdir(folder_path)
    pattern = re.compile(f'^{folder_name}game(\d+)')
    numbers = []
    
    for filename in files:
        match = pattern.match(filename)
        if match:
            numbers.append(int(match.group(1)))
    
    if numbers:
        highest = max(numbers)
        return highest + 1
    else:
        return 1


def convert_node_to_line(node):
    line = "node " + node.descr + " "
    if node.type == 'player':
        line += "player " + node.player + " " + "actions " + " ".join(node.actions) + "\n"
    elif node.type == 'chance':
            action_info = [action + "=" + str(prob) for action, prob in node.action_probs.items()]
            line += "chance actions " + " ".join(action_info) + "\n"
    elif node.type == 'leaf':
            payoff_info = [str(i+1) + "=" + str(node.payoffs[i]) for i in range(len(node.payoffs))]
            line += "leaf payoffs " + " ".join(payoff_info) + "\n"
    else:
        raise Exception("Something went wrong with node['type'], it is invalid", getattr(node, 'type', None))

    return line


def convert_infoset_to_line(infoset):
    line = "infoset " + infoset.descr + " nodes " + " ".join(infoset.node_descrs) + "\n"
    return line



def write(folder_path, specifications, nodes_dict, infs_dict, infs_keys):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    new_number = next_file_number(folder_path,folder_name)
    game_name = "game" + str(new_number)
    unique_id = str(hash(str(specifications) + str(int(time.time()))))
    new_filedir = folder_path + f'/{folder_name}{game_name}.txt'
    print("You can find the game saved under ", new_filedir)

    yaml_str = yaml.dump(specifications, default_flow_style=False, sort_keys=False)

    with open(new_filedir, "w") as file:
        file.write(f"# {game_name}\n")
        file.write(f"# Unique ID: {unique_id}\n")
        file.write("# \n")
        file.write(f"# Number of Nodes: {len(nodes_dict.keys())}\n")
        file.write(f"# Number of Infosets: {len(infs_dict.keys())}\n")
        file.write("# Randomly generated game under the following parameters:\n")
        file.write("# \n")

        for line in yaml_str.splitlines():
            file.write("# " + line + "\n")

        file.write("# \n")
        file.write("# \n")

        for node in nodes_dict.values():
            file.write( convert_node_to_line(node) )
        for inf_key in infs_keys:
            file.write( convert_infoset_to_line(infs_dict[inf_key]) )

    return new_filedir, game_name