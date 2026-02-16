
#%%
import numpy as np
import string
import copy
import colorsys



#%%
def sample_from_simplex(dim, np_rng=None):
    if np_rng is None:
        np_rng = np.random
    x = np_rng.rand(dim)
    x.sort()
    sample = np.diff(np.concatenate(([0], x, [1])))
    
    return sample

#%%
def make_rational(string):
    try:
        float(string)
        return float(string)
    except ValueError:
        nom, denom = string.split('/')
        return int(nom) / int(denom)


# %%
def is_float_or_rational(string):
    try:
        float(string)
        return True
    except ValueError:
        values = string.split('/')
        return len(values) == 2 and all(i.isdigit() for i in values)


def afewletters(k):
    if k <= 26:
        letters = list(string.ascii_lowercase[:k])
    else:
        letters = [string.ascii_lowercase[i % 26] + str( 1 + (i // 26) ) for i in range(k)]
    
    return letters


def change_yaml(standard,changes):
    #Changes the standard yaml file according to the changes dictionary
    new = copy.deepcopy(standard)
    for key, value in changes.items():
        if key == "delete":
            for key_lists in value:
                recursive = new
                for subkey in key_lists[:-1]:
                    recursive = recursive[subkey]
                del recursive[key_lists[-1]]
        elif key not in new:
            raise Exception("Key", key, "not found in standard yaml file")
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                new[key][subkey] = subvalue
        else:    
            new[key] = value
    return new

def logging_frequency(time_not_iter=False):
    if time_not_iter:
        return 10
    else:
        return 10

def generate_amsgrad_color(algo_name):
    """
    Generate a distinct brown/tan/yellow color for AMSGrad configurations.

    AMSGrad configs follow the pattern: ams_a1e{a}_b{b1}_c{b2}
    where:
    - a: -2, -1, 0 (3 values)
    - b1: 0.8, 0.9, 0.99 (3 values)
    - b2: 0.99, 0.999, 0.9999 (3 values)

    This generates 27 distinct colors by varying hue, saturation, and lightness
    in the brown/tan/yellow spectrum.
    """
    # Parse the algorithm name
    parts = algo_name.split('_')

    # Extract parameters
    a_str = parts[1].replace('a1e', '')  # e.g., '-2', '-1', '0'
    b1_str = parts[2].replace('b', '')    # e.g., '0.8', '0.9', '0.99'
    b2_str = parts[3].replace('c', '')    # e.g., '0.99', '0.999', '0.9999'

    # Map to indices (0, 1, 2)
    a_map = {'-2': 0, '-1': 1, '0': 2}
    b1_map = {'0.8': 0, '0.9': 1, '0.99': 2}
    b2_map = {'0.99': 0, '0.999': 1, '0.9999': 2}

    a_idx = a_map[a_str]
    b1_idx = b1_map[b1_str]
    b2_idx = b2_map[b2_str]

    # Generate color in HSL space
    # Hue: 30-50 degrees (browns/tans/yellows)
    # We'll use 3 hue bands for the 3 'a' values
    hue_base = [30, 40, 50]  # Different hue ranges
    hue = (hue_base[a_idx] + b1_idx * 3) / 360.0  # Normalize to [0, 1]

    # Saturation: 0.3-0.8 (vary by b2)
    saturation_values = [0.35, 0.55, 0.75]
    saturation = saturation_values[b2_idx]

    # Lightness: 0.25-0.75 (vary by b1 and b2)
    # Create a 3x3 grid of lightness values
    lightness_grid = [
        [0.30, 0.40, 0.50],  # b2=0.99
        [0.45, 0.55, 0.65],  # b2=0.999
        [0.60, 0.70, 0.80]   # b2=0.9999
    ]
    lightness = lightness_grid[b2_idx][b1_idx]

    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)

    # Convert to hex
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))

    return hex_color

def generate_all_amsgrad_colors():
    """
    Generate color mappings for all 27 AMSGrad configurations.

    Returns a dictionary mapping algorithm names to hex color codes.
    """
    amsgrad_colors = {}
    for a in ['-2', '-1', '0']:
        for b1 in ['0.8', '0.9', '0.99']:
            for b2 in ['0.99', '0.999', '0.9999']:
                algo_name = f"ams_a1e{a}_b{b1}_c{b2}"
                amsgrad_colors[algo_name] = generate_amsgrad_color(algo_name)
    return amsgrad_colors