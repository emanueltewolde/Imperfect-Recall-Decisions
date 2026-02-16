# Graph Generator functions for detection.py

import networkx as nx
import matplotlib.pyplot as plt
import itertools
import random
from copy import deepcopy

def create_2Dgrid(m, n, plot = True):
    #grid graph with m rows n columns
    G = nx.grid_2d_graph(m, n)
    if plot:
        # Optional: Set positions for drawing
        pos = {(i, j): (j, -i) for i in range(m) for j in range(n)}
        return G, pos
    else:
        return G

def create_gnp_graph(n, p, plot = True):
    #Create a random graph according to the Erdős-Rényi-Gilbert G(n,p) model, where n is the number of nodes in the graph and p is the probability with which an edge between any two nodes is created
    G = nx.gnp_random_graph(n, p)
    if plot:
        # Optional: Set positions for drawing
        pos = nx.spring_layout(G, seed=42)
        return G, pos
    else:
        return G

def create_gnm_graph(n, m, plot = True):
    #Create a random graph according to the Erdős-Rényi G(n,m) model, where n is the number of nodes in the graph and m is the number of edges that will be distributed uniformly randomly
    G = nx.gnm_random_graph(n, m)
    if plot:
        # Optional: Set positions for drawing
        pos = nx.spring_layout(G, seed=42)
        return G, pos
    else:
        return G

def graph_generator(graph_specs, plot = True):
    if graph_specs["distribution"] == "grid": 
        return create_2Dgrid(*graph_specs["parameter_list"], plot = plot)
    elif graph_specs["distribution"] == "gnp":
        return create_gnp_graph(*graph_specs["parameter_list"], plot = plot)
    elif graph_specs["distribution"] == "gnm":
        return create_gnm_graph(*graph_specs["parameter_list"], plot = plot)
    else:
        raise ValueError("Unknown distribution type")

def create_line_graph(k):
    G = nx.path_graph(k)
    return G

def create_cycle_graph(k):
    G = nx.cycle_graph(k)
    return G

def create_clique_graph(k):
    G = nx.complete_graph(k)
    return G

def create_star_graph(k):
    #create a star with a center with degree k
    G = nx.star_graph(k)
    return G

pattern_generators = {
        "line": create_line_graph,
        "cycle": create_cycle_graph,
        "clique": create_clique_graph,
        "star": create_star_graph
    }

pattern_priority = {"clique": 12, "star": 8, "cycle": 4, "line": 0}

# Sorting key function for lexicographic comparison
def sort_key(desired_pattern):
    return (pattern_priority[desired_pattern[0]], desired_pattern[1])

def sort_in_decreasing_priority(desired_patterns):
    return sorted(desired_patterns, key=sort_key, reverse=True)


def randomly_generate_desired_patterns_from_bounds(num_pattern_bounds, size_bounds):
    #Generate a list of random patterns.
    desired_patterns = []
    num_patterns = random.randint(num_pattern_bounds[0], num_pattern_bounds[1])
    for _ in range(num_patterns):
        size = random.randint(size_bounds[0], size_bounds[1])
        shape = random.choice(list(pattern_generators.keys()))
        desired_patterns.append([shape, size])
    
    return sort_in_decreasing_priority(desired_patterns)

def randomly_generate_desired_patterns_from_numnodes(max_num_nodes, size_bounds):
    #Generate a list of random patterns.
    desired_patterns = []
    num_nodes = 0
    assert size_bounds[0] >= 1
    while num_nodes <= max_num_nodes - 3:
        size = random.randint(size_bounds[0], size_bounds[1])
        shape = random.choice(list(pattern_generators.keys()))
        desired_patterns.append([shape, size])
        if shape == "star":
            size += 1
        num_nodes += size
    
    return sort_in_decreasing_priority(desired_patterns)
    

def find_subgraphH(G, H):
    h_size = len(H.nodes())
    results = []
    # For each possible set of nodes that could form the pattern
    for nodes in itertools.combinations(G.nodes(), h_size):
        if nodes == ((0,0), (0,1), (0,2), (1,0), (1,1), (1,2)):
            print("hi")
        if nx.is_isomorphic(G.subgraph(nodes), H):
            results.append(nodes)
    
    return results

def find_allHpatterns(G, H):
    h_size = len(H.nodes())
    h_edges = len(H.edges())
    patterns = []
    # For each possible set of nodes that could form the pattern
    for nodes in itertools.combinations(G.nodes(), h_size):
        subgraph = G.subgraph(nodes)
        # Check if subgraph contains at least all edges that would be in H
        # (it might have more edges, which is fine)
        if len(subgraph.edges()) >= h_edges:
            # Try to find a correspondence between nodes in subgraph and H
            # that would make subgraph contain H
            for h_nodes_perm in itertools.permutations(H.nodes()):
                # Create a mapping from H nodes to subgraph nodes
                mapping = dict(zip(h_nodes_perm, nodes))
                
                # Check if all edges in H exist in the subgraph under this mapping
                all_edges_exist = True
                edge_list = []
                for u,v in H.edges():
                    edge_list.append((mapping[u], mapping[v]))
                    if not subgraph.has_edge(mapping[u], mapping[v]):
                        all_edges_exist = False
                        break
                
                if all_edges_exist:
                    patterns.append([nodes,edge_list])
                    break  # Found a valid mapping, no need to check other permutations
    
    return patterns

def list_all_patterncombis(Graph, desired_patterns, concise_and_list=False):
    #Desired patterns is a list of lists [name_string, parameter_number] such name_string is in the dictionary pattern_generators at the top
    desired_patterns = sort_in_decreasing_priority(desired_patterns)
    all_pattern_combis = []
    num_combi = len(desired_patterns)

    def grow(G, partial):
        current_index = len(partial)
        if current_index == num_combi:
            all_pattern_combis.append(partial)
        else:
            next_des = desired_patterns[ current_index ]
            pattern = pattern_generators[next_des[0]](next_des[1])
            found = find_allHpatterns(G,pattern)
            for selected_pattern in found:
                newpartial = deepcopy(partial)
                if concise_and_list:
                    newpartial.append(list(selected_pattern[0]))
                else:
                    newpartial.append(selected_pattern)
                Gprime = deepcopy(G)
                Gprime.remove_nodes_from(selected_pattern[0])
                grow(Gprime, newpartial)

    
    grow(Graph, [])
    return all_pattern_combis

def find_oneHpattern(G, H):
    h_size = len(H.nodes())
    h_edges = len(H.edges())
    
    all_combinations_random_order = list(itertools.combinations(G.nodes(), h_size))
    random.shuffle(all_combinations_random_order)
    # For each possible set of nodes that could form the pattern
    for nodes in all_combinations_random_order:
        subgraph = G.subgraph(nodes)
        # Check if subgraph contains at least all edges that would be in H
        # (it might have more edges, which is fine)
        if len(subgraph.edges()) >= h_edges:
            # Try to find a correspondence between nodes in subgraph and H
            # that would make subgraph contain H
            for h_nodes_perm in itertools.permutations(H.nodes()):
                # Create a mapping from H nodes to subgraph nodes
                mapping = dict(zip(h_nodes_perm, nodes))
                
                # Check if all edges in H exist in the subgraph under this mapping
                all_edges_exist = True
                edge_list = []
                for u,v in H.edges():
                    edge_list.append((mapping[u], mapping[v]))
                    if not subgraph.has_edge(mapping[u], mapping[v]):
                        all_edges_exist = False
                        break
                
                if all_edges_exist:
                    return [nodes,edge_list] # Found a valid mapping, no need to check other permutations or other nodes subsets
    return False

def select_patterns_randomly(Graph, desired_patterns, concise_and_list=False):
    #Desired patterns is a list of lists [name_string, parameter_number] such name_string is in the dictionary pattern_generators at the top
    #Caution: The order in desired_patterns matters for the random selection
    desired_patterns = sort_in_decreasing_priority(desired_patterns)
    G = deepcopy(Graph)
    sampled_patterns = []
    for des_pat in desired_patterns:
        pattern = pattern_generators[des_pat[0]](des_pat[1])
        selected_pattern = find_oneHpattern(G,pattern)
        if selected_pattern == False:
            # print(des_pat, " could not be found among the remaining subgraph")
            return False
        else:
            if concise_and_list:
                sampled_patterns.append(list(selected_pattern[0]))
            else:
                sampled_patterns.append(selected_pattern)
            # Remove the selected pattern from G
            G.remove_nodes_from(selected_pattern[0])

    return sampled_patterns

def randomly_sample_n_pattern_combis(graph, num_samples, num_pattern_bounds, pattern_size_bounds, valuation_bounds, concise_and_list=False):
    pattern_combis = []
    pattern_combis_set = set()
    valuation_combis = []
    num_combis = 0
    for i in range(200):
        desired_patterns = randomly_generate_desired_patterns_from_bounds(num_pattern_bounds, pattern_size_bounds)
        sampled_patterns = select_patterns_randomly(graph, desired_patterns, concise_and_list=concise_and_list)
        if sampled_patterns:
            sampled_patterns_set = frozenset(frozenset(pattern) for pattern in sampled_patterns)
            if sampled_patterns_set not in pattern_combis_set:
                pattern_combis_set.add(sampled_patterns_set)
                valuations = [random.randint(valuation_bounds[0], valuation_bounds[1]) for _ in range(len(sampled_patterns))]
                pattern_combis.append(sampled_patterns)
                valuation_combis.append(valuations)
                num_combis += 1
                if num_combis >= num_samples:
                    return [pattern_combis, valuation_combis]

    print(f"Could only find {num_combis} out of the desired {num_samples} pattern combinations. Returning empty lists.")
    return [[], []]

def create_graph_and_patterncombis(specs):
    graph = graph_generator(specs["graph"], plot = False)
    if specs["communities"]["method"] == "enumerate":
        desired_patterns = specs["communities"]["community_list"]
        return graph, list_all_patterncombis(graph, desired_patterns, concise_and_list=True)
    elif specs["communities"]["method"] == "random":
        return graph, randomly_sample_n_pattern_combis(graph, specs["communities"]["num_samples"], specs["communities"]["num_communities_bounds"], specs["communities"]["community_size_bounds"], specs["communities"]["valuation_bounds"], concise_and_list=True)
    else:
        raise ValueError("Unknown community detection method")



def individually_plot_patterns_in_graph(G, pos, patterns, title_prefix="Graph with Subgraphs", 
                             highlight_color='red', base_color='lightblue', 
                             highlight_width=3.0, base_width=1.0):
    if not patterns:
        print("No subgraphs found to highlight")
        return
    
    for i, pattern in enumerate(patterns):
        plt.figure(figsize=(8, 6))
        
        subgraph_nodes, subgraph_edges = pattern
        
        all_nodes = list(G.nodes())
        non_subgraph_nodes = [n for n in all_nodes if n not in subgraph_nodes]
        
        all_edges = list(G.edges())
        non_subgraph_edges = [e for e in all_edges if e not in subgraph_edges]
        
        nx.draw_networkx_nodes(G, pos, nodelist=non_subgraph_nodes,
                              node_color=base_color, node_size=500, alpha=0.7)
        nx.draw_networkx_edges(G, pos, edgelist=non_subgraph_edges,
                              width=base_width, alpha=0.5)
        
        nx.draw_networkx_nodes(G, pos, nodelist=subgraph_nodes,
                              node_color=highlight_color, node_size=500)
        nx.draw_networkx_edges(G, pos, edgelist=subgraph_edges,
                              width=highlight_width, edge_color=highlight_color)
        
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title(f"{title_prefix} - Subgraph {i+1}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def plot_patterns_in_graph(G, pos, patterns, title_prefix="Graph with Patterns", 
                             highlight_color='red', base_color='lightblue', 
                             highlight_width=3.0, base_width=1.0):
    
    pattern_nodes = set()
    pattern_edges = set()
    for pattern in patterns:
        pattern_nodes.update(pattern[0])
        pattern_edges.update(pattern[1])

    plt.figure(figsize=(8, 6))
    
    all_nodes = list(G.nodes())
    non_subgraph_nodes = [n for n in all_nodes if n not in pattern_nodes]
    
    all_edges = list(G.edges())
    non_subgraph_edges = [e for e in all_edges if e not in pattern_edges]
    
    nx.draw_networkx_nodes(G, pos, nodelist=non_subgraph_nodes,
                            node_color=base_color, node_size=500, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=non_subgraph_edges,
                            width=base_width, alpha=0.5)
    
    nx.draw_networkx_nodes(G, pos, nodelist=pattern_nodes,
                            node_color=highlight_color, node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=pattern_edges,
                            width=highlight_width, edge_color=highlight_color)
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title(f"{title_prefix}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n, p = 10, 0.5
    G, pos = create_gnp_graph(n, p)

    patterncombis = list_all_patterncombis(G, [["line", 2], ["cycle", 3], ["star", 4]])
    
    print(len(patterncombis))
    for combo in patterncombis:
        plot_patterns_in_graph(G, pos, combo, f"Graph")
