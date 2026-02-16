# All plotting functions for the convergence behavior and time performance of the algorithms

import re
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import gurobi_logtools as glt
import pandas as pd
import subprocess

from modules.utils import logging_frequency, generate_all_amsgrad_colors

if __name__ == "__main__":
    
    def logging_frequency(time_not_iter=False):
        if time_not_iter:
            return 0.02
        else:
            return 25

global_tolerance = 1e-6
global_maxiter = 6000

script_dir = os.path.dirname(os.path.abspath(__file__))
library_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Generate all AMSGrad colors (27 configurations)
_amsgrad_colors = generate_all_amsgrad_colors()

algorithm_colors = {
    # Regret Matching family (reds/oranges)
    "rm": "#e41a1c",    # bright red
    "rm+": "#ff7f00",   # orange
    "prm": "#fb9a99",   # light red
    "prm+": "#fdbf6f",  # light orange

    # Projected Gradient Descent family (blues)
    "pgd_1e-3": "#deebf7",  # very light blue
    "pgd_1e-2": "#6baed6",  # light blue
    "pgd_1e-1": "#377eb8",  # medium blue
    "pgd_1e0": "#08519c",   # dark blue

    # Optimistic Gradient Descent family (purples)
    "optgd_1e-3": "#cbc9e2",  # very light purple
    "optgd_1e-2": "#9e9ac8",  # light purple
    "optgd_1e-1": "#8856a7",  # medium-light purple
    "optgd_1e0": "#4d004b",   # deep purple/plum

    # Gurobi (green)
    "gurobi": "#4daf4a"     # green
}

# Add all 27 AMSGrad colors
algorithm_colors.update(_amsgrad_colors)

best_colors = {
    "rm": ["#AA3377",'o'],
    "rm+": ["#EE6677",'s'],
    "prm": ["#CCBB44",'D'],
    "prm+": ["#E69F00",'+'],
    "pgd": ["#87D0E9",'x'],
    "optgd": ["#155FAA",'^'],
    "ams": ["#10D3AC",'*']
    }

font = {
    "rm": r'$\mathtt{RM}$',
    "rm+": r'$\mathtt{RM^+}$',
    "prm": r'$\mathtt{PRM}$',
    "prm+": r'$\mathtt{PRM^+}$',
    "pgd": r'$\mathtt{GD}$',
    "optgd": r'$\mathtt{OGD}$',
    "ams": r'$\mathtt{AMS}$'
}
# \math


def extract_from_my_logging_format(logfile):
    results = []

    with open(logfile, 'r') as file:
        log_text = file.read()


    lines = log_text.strip().split('\n')
    
    pattern = r'iter (\d+) \| time ([\d.e-]+) \| value ([\d.e-]+) \| gap ([\d.e-]+)'
    
    for line in lines:
        if ' -- ' in line:
            parts = line.split(' -- ', 1)
            content = parts[1].strip()
            
            # Skip comments (lines with hash)
            if content.startswith('#'):
                continue
                
            match = re.search(pattern, content)
            if match:
                iter_num = int(match.group(1))
                time_val = float(match.group(2))
                value_val = float(match.group(3))
                gap_val = float(match.group(4))
                
                results.append({
                    'iter': iter_num,
                    'time': time_val,
                    'value': value_val,
                    'gap': gap_val
                })
    
    return results


def plot_one_seedrun(multiple_results, max_iter=False):

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for algo_name, results in multiple_results.items():
        iterations = [result['iter'] for result in results if max_iter and result['iter'] <= max_iter]
        values = [result['value'] for result in results if max_iter and result['iter'] <= max_iter]
        plt.plot(iterations, values, label=algo_name)
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.title('Convergence over Iterations')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    for algo_name, results in multiple_results.items():
        iterations = [result['iter'] for result in results if max_iter and result['iter'] <= max_iter]
        gaps = [result['gap'] for result in results if max_iter and result['iter'] <= max_iter]
        plt.plot(iterations, gaps, label=algo_name)
    plt.xlabel('Iterations')
    plt.ylabel('Gap')
    plt.yscale('log')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


def extract_from_logging(logfile, gurobi=False):
    if gurobi:
        return parse_gurobi_logging(logfile)
    else:
        return extract_from_my_logging_format(logfile)

def parse_gurobi_logging(log_file):

    log_dir = os.path.dirname(log_file)
    # Create output filename in same directory
    gurobi_file = os.path.join(log_dir, "extracted_gurobi_logging.txt")

    with open(log_file, 'r') as file:
        log_text = file.read()
        lines = log_text.strip().split('\n')
        with open(gurobi_file, 'w') as f:
            for line in lines:
                assert ' --' in line            
                parts = line.split(' --', 1)
                content = parts[1]

                if not content.startswith(" #") and len(content) > 0:            
                    f.write(content[1:] + '\n')

    table = glt.parse(gurobi_file).progress("nodelog")
    if 'Incumbent' not in table.columns:
        # print(f"Warning: there seems to be no progress reported in {log_file}. We will return None instead.")
        return []
    else:
        obj_values = table['Incumbent'].tolist()
        times = table['Time'].tolist()
        return [ {'time': time, 'value': value} for value, time in zip(obj_values,times) ]

def all_experiments_for_game(log_folder, algo_exclusions=[]):
    all_results = {}
    seen_combos = []
    nice_name = None
    game_name = "definitely not this name"
    name_updates = 0
    for root, dirs, files in os.walk(log_folder):
        for file in files:
            if '__' not in file:
                continue
            info = file.split('__')
            if not ( 'game' in info[0] and info[1].startswith('algo-') ):
                continue
            
            full_path = os.path.join(root, file)
            
            new_game_name = info[0]
            if new_game_name != game_name:
                game_name = new_game_name
                name_updates += 1

                nice_name = root.split('runs/')[1]
                # nice_name = game_name.split('game')[0].upper() + game_name.split('game')[1]
                # nice_name = os.path.basename(os.path.dirname(os.path.dirname(root))).capitalize() + nice_name

            algo = info[1].replace('algo-', '')
            if algo in algo_exclusions:
                continue

            if algo == 'gurobi':
                if info[-1].startswith('blockedreruns'):
                    continue
                
                assert algo not in all_results
                result = extract_from_logging(full_path, gurobi=True)
                all_results[algo] = []
                for entry in result:
                    entry["seed"] = "None"
                    all_results[algo].append(entry)
            else:
                assert info[2].startswith('seed-')
                seed = info[2].replace('seed-', '')
                assert seed.isdigit()
                seed = int(seed)
                assert [algo,seed] not in seen_combos
                seen_combos.append([algo,seed])

                if algo not in all_results:
                    all_results[algo] = []

                result = extract_from_logging(full_path, gurobi=False)
                for entry in result:
                    entry["seed"] = seed
                    all_results[algo].append(entry)
            
    assert name_updates == 1
    for algo, results in all_results.items():
        all_results[algo] = pd.DataFrame(results)

    return all_results, nice_name

def plot_a_game(logfolder, show=True, output_dir=None, confidencepercentile=0.25, which_plots = {'value-time': True, 'gap-iter': True, 'value-iter': False, 'gap-time': False}, max_length = {'time': 7200, 'iter': 2000}, algo_exclusions = []):
    
    assert os.path.exists(logfolder), f"Directory {os.path.dirname(logfolder)} does not exist"
    all_data, game_name = all_experiments_for_game(logfolder, algo_exclusions=algo_exclusions)

    num_plots = sum(which_plots.values())
    plt.figure(figsize=(12, 4*num_plots))

    subplot_id = 1

    for string, val in which_plots.items():
        if not val:
            continue
        
        yaxis_descr, xaxis_descr = string.split('-')
        add_gurobi = (yaxis_descr == 'value' and xaxis_descr == 'time')
        
        freq = logging_frequency(time_not_iter= (xaxis_descr == 'time') )

        #First get the lowest value of yaxis_descr across all algorithms
        assert (yaxis_descr == 'value' or yaxis_descr == 'gap')
        
        y_worst = 0
        updated = False
        
        for algo,data in all_data.items():
            if algo == 'gurobi' and not add_gurobi:
                continue
            if xaxis_descr not in data.columns:
                # print(f"Warning: {xaxis_descr} was not found in data for algorithm {algo}. Check if this is because the game was to big to be built in the first place? Skipping this one.")
                continue
            
            updated = True

            if yaxis_descr == 'gap': 
                y_worst = max( data[yaxis_descr].max(), y_worst )
            else:
                y_worst = min( data[yaxis_descr].min(), y_worst )
        
        if not updated:
            print(f"Warning: {yaxis_descr} was not found in data for any algorithm. Maybe the game was to big to build in the first place? Skipping this one.")
            continue

        plt.subplot(num_plots, 1, subplot_id)
        subplot_id += 1
        for algo,data in all_data.items():

            if algo == 'gurobi' and not add_gurobi:
                continue
            
            if xaxis_descr not in data.columns:
                print(f"Warning: {xaxis_descr} was not found in data for algorithm {algo}. In the case of gurobi, maybe the game was to big to get to branch and bound in the first place? Adding a bad plot for this one.")
                data = pd.DataFrame([{xaxis_descr: 0, yaxis_descr: y_worst, 'seed': "seed"}, {xaxis_descr: max_length[xaxis_descr], yaxis_descr: y_worst, 'seed': "seed"}])              

                # continue

            max_x_found = data[xaxis_descr].max()
            first_multiple_geq_max = round(math.ceil(max_x_found / freq)  * freq, 4)

            stats = pd.DataFrame()
            max_found_x_per_seed = data.groupby('seed')[xaxis_descr].max()
            for seed, max_x_seed in max_found_x_per_seed.items():
                local_data = data[data['seed'] == seed][[xaxis_descr, yaxis_descr, 'seed']]
                stats = standardize_and_expand(stats, local_data, seed, xaxis_descr, yaxis_descr, freq, y_worst, max_x_seed, first_multiple_geq_max)

            if max_length[xaxis_descr]:
                limit = min( max_length[xaxis_descr], first_multiple_geq_max )
            else:
                limit = first_multiple_geq_max
            stats = stats[stats[xaxis_descr] <= limit]

            lower_percentile = lambda x: np.percentile(x, confidencepercentile * 100)
            upper_percentile = lambda x: np.percentile(x, (1 - confidencepercentile) * 100)
            
            stats = stats.groupby(xaxis_descr)[yaxis_descr].agg([
                ('median', 'median'),
                ('lower', lower_percentile),
                ('upper', upper_percentile)
            ]).reset_index()
            
            plt.plot(stats[xaxis_descr], stats['median'], color=algorithm_colors[algo], linewidth=2, label=algo)
            
            plt.fill_between(
                stats[xaxis_descr],
                stats['lower'],
                stats['upper'],
                color=algorithm_colors[algo], alpha=0.3)
            
        plt.xlabel(xaxis_descr.capitalize())
        plt.ylabel(yaxis_descr.capitalize())
        if yaxis_descr == 'gap':
            plt.yscale('log')
            plt.ylim(bottom=global_tolerance)
            plt.minorticks_off()
        plt.title(f'{yaxis_descr.capitalize()} progress in {game_name} across seeds.')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

    if output_dir is not None:
        output_path = os.path.join(output_dir, f"{game_name.replace('/', '_')}__confidencepercentile-{confidencepercentile}_max_lengths-{max_length['time']}s{max_length['iter']}iter.pdf")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            response = input(f"File {output_path} already exists. Do you want to override it? (y/n): ")
        else:
            response = 'y'
        
        if response.lower() in ['y', 'yes']:
            plt.savefig(output_path, bbox_inches='tight', dpi=300, format='pdf')
            print(f"Figure saved to {output_path}")
        else:
            print(f"File not saved. Skipping.")

    if show:
        plt.show()
    else:
        plt.close()

def standardize_and_expand(stats, local_data, seed, xaxis_descr, yaxis_descr, freq, worst_y, max_x_seed, end_multiple):
    # Verify that local_data is sorted by xaxis_descr values in ascending order
    assert all(local_data[xaxis_descr].iloc[i] <= local_data[xaxis_descr].iloc[i+1] for i in range(len(local_data)-1)), f"Data for seed {seed} is not monotonically increasing in {xaxis_descr}"

    min_x = local_data[xaxis_descr].min()

    #pad at the beginning if the algo took too long (in time) to get started
    for x_val in np.arange(0, min_x, freq):
        new_row = {xaxis_descr: x_val, yaxis_descr: worst_y, 'seed': seed}
        stats = pd.concat([stats, pd.DataFrame([new_row])], ignore_index=True)

    
    first = local_data.iloc[0].copy()
    if min_x % freq == 0:
        stats = pd.concat([stats, pd.DataFrame([first])], ignore_index=True)
        first_multiple_geq_min = min_x + freq
    else:
        first_multiple_geq_min = round(math.ceil(min_x / freq) * freq, 4)  
    
    #iterates through all multiples of freq at least as large as min_x and strictly smaller than max_x_seed
    for x_val in np.arange(first_multiple_geq_min , max_x_seed, freq):
        # Find the row with the highest x-value that's less than or equal to x_val
        x_val = round(x_val, 4)
        last_idx = local_data[local_data[xaxis_descr] <= x_val].index[-1]  
        closest_row = local_data.loc[last_idx]
        next_row = local_data.loc[last_idx + 1]
        y_val = linear_interpolate(closest_row[xaxis_descr], next_row[xaxis_descr], x_val, closest_row[yaxis_descr], next_row[yaxis_descr])
        new_row = {xaxis_descr: x_val, yaxis_descr: y_val, 'seed': seed}
        stats = pd.concat([stats, pd.DataFrame([new_row])], ignore_index=True)
    

    #iterates through all multiples of freq at least as large as max_x_seed and at most as large as the first multiple larger from max_x onwards (inclusive)
    last = local_data.iloc[-1]
    first_multiple_geq_max_seed = round(math.ceil(max_x_seed / freq)  * freq, 4)
    for x_val in np.arange(first_multiple_geq_max_seed, end_multiple + freq/2, freq):
        x_val = round(x_val, 4)
        copy = last.copy()
        copy[xaxis_descr] = x_val
        stats = pd.concat([stats, pd.DataFrame([copy])], ignore_index=True)

    return stats

def timebest_gd(all_d): 
    best_median_time = {'pgd':float('inf'), 'optgd':float('inf')}
    best_algo = {'pgd':None, 'optgd':None}

    for algo, data in all_d.items():
        if 'iter' not in data.columns:
            # Skip algorithms without iteration data
            continue
        for gd in ['pgd', 'optgd']:
            if gd in algo:    
                # Group by seed and find the first time that gap goes below tolerance
                times_to_convergence = []
                for seed in data['seed'].unique():
                    seed_data = data[data['seed'] == seed]
                    # Never reached tolerance - use the maximum time
                    max_time = seed_data['time'].max()
                    # Get the row with the maximum time
                    max_time_row = seed_data[seed_data['time'] == max_time].iloc[0]
                    # Check if iter count at max_time is less than global_maxiter
                    if max_time_row['iter'] +1 < global_maxiter:
                        times_to_convergence.append(max_time)
                    else:
                        # If it reached or exceeded global_maxiter, it didn't converge
                        times_to_convergence.append(float('inf'))
                
                median_time = np.median(times_to_convergence)
                if median_time < best_median_time[gd]:
                    best_median_time[gd] = median_time
                    best_algo[gd] = algo           
    return best_algo['pgd'], best_algo['optgd']

def plot_game_best(logfolder, show=True, output_dir=None, confidencepercentile=0.25, add_value_iter=True, max_length = {'iter': 2000}, algo_exclusions = [], surpress = False):
    
    assert os.path.exists(logfolder), f"Directory {os.path.dirname(folder)} does not exist"
    algo_exclusions.append('gurobi')
    all_data, game_name = all_experiments_for_game(logfolder, algo_exclusions=algo_exclusions)
    
    gap_worst = 0
    if add_value_iter: value_worst = float('inf')

    for algo,data in all_data.items():
        if 'iter' not in data.columns:
            # print(f"Warning: {xaxis_descr} was not found in data for algorithm {algo}. Check if this is because the game was to big to be built in the first place? Skipping this one.")
            continue
        
        gap_worst = max( data['gap'].max(), gap_worst )
        if add_value_iter: value_worst = min( data['value'].min(), value_worst )

    # Set font to Times
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'text.usetex': True
    })

    if add_value_iter:
        fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(4, 4))
    else:
        fig, ax2 = plt.subplots(figsize=(6, 2.5))

    bestpgd, bestoptgd, bestams = None, None, None
    freq = logging_frequency(time_not_iter= False)
    
    for algo,data in all_data.items():
        if algo == 'gurobi':
            continue

        if 'iter' not in data.columns:
            data = pd.DataFrame([{'iter': 0, 'gap': gap_worst, 'seed': "seed"}, {'iter': max_length['iter'], 'gap': gap_worst, 'seed': "seed"}]) 
                
        max_x_found = data['iter'].max()
        first_multiple_geq_max = round(math.ceil(max_x_found / freq)  * freq, 4)

        stats = pd.DataFrame()
        max_found_x_per_seed = data.groupby('seed')['iter'].max()
        for seed, max_x_seed in max_found_x_per_seed.items():
            local_data = data[data['seed'] == seed][['iter', 'gap', 'seed']]
            stats = standardize_and_expand(stats, local_data, seed, 'iter', 'gap', freq, gap_worst, max_x_seed, first_multiple_geq_max)

        if max_length['iter']:
            limit = min( max_length['iter'], first_multiple_geq_max )
        else:
            limit = first_multiple_geq_max
        stats = stats[stats['iter'] <= limit]

        lower_percentile = lambda x: np.percentile(x, confidencepercentile * 100)
        upper_percentile = lambda x: np.percentile(x, (1 - confidencepercentile) * 100)
        
        stats = stats.groupby('iter')['gap'].agg([
            ('median', 'median'),
            ('lower', lower_percentile),
            ('upper', upper_percentile)
        ]).reset_index()
        
        below_tolerance = stats[stats['median'] <= global_tolerance]
        
        # First determine where to cut off the median line (when median reaches tolerance)
        if not below_tolerance.empty:
            first_below = below_tolerance.iloc[0]
            cutoff_index = stats[stats['iter'] == first_below['iter']].index[0]
            plot_stats_median = stats.iloc[:cutoff_index+1]
        else:
            cutoff_index = None
            plot_stats_median = stats

        # Determine where to cut off the upper bound (when upper reaches tolerance)
        below_tolerance_upper = stats[stats['upper'] <= global_tolerance]
        if not below_tolerance_upper.empty:
            first_below_upper = below_tolerance_upper.iloc[0]
            cutoff_index_upper = stats[stats['iter'] == first_below_upper['iter']].index[0]
            plot_stats_fill = stats.iloc[:cutoff_index_upper+1]
        else:
            cutoff_index_upper = None
            plot_stats_fill = stats

        if 'pgd' in algo:
            if bestpgd is None:
                bestpgd = {'algo': algo, 'gap_median':plot_stats_median, 'gap_fill':plot_stats_fill}
            else:
                if plot_stats_median['median'].iloc[-1] <= global_tolerance:
                    better = (bestpgd['gap_median']['median'].iloc[-1] > global_tolerance) or (plot_stats_median['iter'].iloc[-1] < bestpgd['gap_median']['iter'].iloc[-1])
                else:
                    better = plot_stats_median['median'].iloc[-1] < bestpgd['gap_median']['median'].iloc[-1]
                if better:
                    bestpgd = {'algo': algo, 'gap_median':plot_stats_median, 'gap_fill':plot_stats_fill}
                else:
                    continue
        elif 'optgd' in algo:
            if bestoptgd is None:
                bestoptgd = {'algo': algo, 'gap_median':plot_stats_median, 'gap_fill':plot_stats_fill}
            else:
                if plot_stats_median['median'].iloc[-1] <= global_tolerance:
                    better = (bestoptgd['gap_median']['median'].iloc[-1] > global_tolerance) or (plot_stats_median['iter'].iloc[-1] < bestoptgd['gap_median']['iter'].iloc[-1])
                else:
                    better = plot_stats_median['median'].iloc[-1] < bestoptgd['gap_median']['median'].iloc[-1]
                if better:
                    bestoptgd = {'algo': algo, 'gap_median':plot_stats_median, 'gap_fill':plot_stats_fill}
                else:
                    continue
        elif 'ams' in algo:
            if bestams is None:
                bestams = {'algo': algo, 'gap_median':plot_stats_median, 'gap_fill':plot_stats_fill}
            else:
                if plot_stats_median['median'].iloc[-1] <= global_tolerance:
                    better = (bestams['gap_median']['median'].iloc[-1] > global_tolerance) or (plot_stats_median['iter'].iloc[-1] < bestams['gap_median']['iter'].iloc[-1])
                else:
                    better = plot_stats_median['median'].iloc[-1] < bestams['gap_median']['median'].iloc[-1]
                if better:
                    bestams = {'algo': algo, 'gap_median':plot_stats_median, 'gap_fill':plot_stats_fill}
                else:
                    continue
        else:
            ax1.plot(plot_stats_median['iter'], plot_stats_median['median'], 
                        color=best_colors[algo][0], marker=best_colors[algo][1], markevery=5, linewidth=2, label=font[algo])

            ax1.fill_between(
                plot_stats_fill['iter'],
                plot_stats_fill['lower'],
                plot_stats_fill['upper'],
                color=best_colors[algo][0], alpha=0.3)
                    
        if not add_value_iter:
            continue

        if 'iter' not in data.columns:
            data = pd.DataFrame([{'iter': 0, 'value': value_worst, 'seed': "seed"}, {'iter': max_length['iter'], 'value': value_worst, 'seed': "seed"}]) 

        stats = pd.DataFrame()
        for seed, max_x_seed in max_found_x_per_seed.items():
            local_data = data[data['seed'] == seed][['iter', 'value', 'seed']]
            stats = standardize_and_expand(stats, local_data, seed, 'iter', 'value', freq, value_worst, max_x_seed, first_multiple_geq_max)

        stats = stats[stats['iter'] <= limit]
        
        stats = stats.groupby('iter')['value'].agg([
            ('median', 'median'),
            ('lower', lower_percentile),
            ('upper', upper_percentile)
        ]).reset_index()
        
        if cutoff_index is not None:
            plot_stats_median = stats.iloc[:cutoff_index+1]
        else:
            plot_stats_median = stats
        
        if cutoff_index_upper is not None:
            plot_stats_fill = stats.iloc[:cutoff_index_upper+1]
        else:
            plot_stats_fill = stats

        if 'pgd' in algo:
            bestpgd.update({'value_median':plot_stats_median, 'value_fill':plot_stats_fill})
        elif 'optgd' in algo:
            bestoptgd.update({'value_median':plot_stats_median, 'value_fill':plot_stats_fill})
        elif 'ams' in algo:
            bestams.update({'value_median':plot_stats_median, 'value_fill':plot_stats_fill})
        else:
            ax2.plot(plot_stats_median['iter'], plot_stats_median['median'], 
                        color=best_colors[algo][0], marker=best_colors[algo][1], markevery=5, linewidth=2, label=font[algo])

            ax2.fill_between(
                plot_stats_fill['iter'],
                plot_stats_fill['lower'],
                plot_stats_fill['upper'],
                color=best_colors[algo][0], alpha=0.3)
        
    
    # BestPGD and BestOPTGD and BestAMS
    ax1.plot(bestpgd['gap_median']['iter'], bestpgd['gap_median']['median'], 
                color=best_colors['pgd'][0], marker=best_colors['pgd'][1], markevery=5, linewidth=2, label=font['pgd'])
    ax1.fill_between(
        bestpgd['gap_fill']['iter'],
        bestpgd['gap_fill']['lower'],
        bestpgd['gap_fill']['upper'],
        color=best_colors['pgd'][0], alpha=0.3)
    
    ax1.plot(bestoptgd['gap_median']['iter'], bestoptgd['gap_median']['median'], 
                color=best_colors['optgd'][0], marker=best_colors['optgd'][1], markevery=5, linewidth=2, label=font['optgd'])
    ax1.fill_between(
        bestoptgd['gap_fill']['iter'],
        bestoptgd['gap_fill']['lower'],
        bestoptgd['gap_fill']['upper'],
        color=best_colors['optgd'][0], alpha=0.3)

    ax1.plot(bestams['gap_median']['iter'], bestams['gap_median']['median'], 
                color=best_colors['ams'][0], marker=best_colors['ams'][1], markevery=5, linewidth=2, label=font['ams'])
    ax1.fill_between(
        bestams['gap_fill']['iter'],
        bestams['gap_fill']['lower'],
        bestams['gap_fill']['upper'],
        color=best_colors['ams'][0], alpha=0.3)
    
    ax2.plot(bestpgd['value_median']['iter'], bestpgd['value_median']['median'], 
                color=best_colors['pgd'][0], marker=best_colors['pgd'][1], markevery=5, linewidth=2, label=font['pgd'])
    ax2.fill_between(
        bestpgd['value_fill']['iter'],
        bestpgd['value_fill']['lower'],
        bestpgd['value_fill']['upper'],
        color=best_colors['pgd'][0], alpha=0.3)
        
    ax2.plot(bestoptgd['value_median']['iter'], bestoptgd['value_median']['median'], 
                color=best_colors['optgd'][0], marker=best_colors['optgd'][1], markevery=5, linewidth=2, label=font['optgd'])
    ax2.fill_between(
        bestoptgd['value_fill']['iter'],
        bestoptgd['value_fill']['lower'],
        bestoptgd['value_fill']['upper'],
        color=best_colors['optgd'][0], alpha=0.3)

    ax2.plot(bestams['value_median']['iter'], bestams['value_median']['median'], 
                color=best_colors['ams'][0], marker=best_colors['ams'][1], markevery=5, linewidth=2, label=font['ams'])
    ax2.fill_between(
        bestams['value_fill']['iter'],
        bestams['value_fill']['lower'],
        bestams['value_fill']['upper'],
        color=best_colors['ams'][0], alpha=0.3)
    
    # with open('./plots/game_infos.json', 'r') as f:
    #     name = json.load(f)[game_name]['new_descr']
    name = game_name.replace('/', '_')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('KKT Gap')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=global_tolerance)
    ax1.grid(True, linestyle='--', alpha=0.7)

    if add_value_iter:
        if not surpress:
            ax2.set_ylabel('Value')
        ax2.set_xticklabels([])
        ax2.set_title(f'{name}')
        ax2.legend(ncol=2, loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()



    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{name}__confidencepercentile-{confidencepercentile}_max_iter-{max_length['iter']}.pdf")
        # Check if file already exists
        # if os.path.exists(output_path):
        #     response = input(f"File {output_path} already exists. Do you want to override it? (y/n): ")
        # else:
        #     response = 'y'
        response = 'y'
        if response.lower() in ['y', 'yes']:
            plt.savefig(output_path, bbox_inches='tight', dpi=300, format='pdf')
            try:
                # Use pdfcrop to crop the PDF (assumes pdfcrop is installed)
                if surpress:
                    left = -10.2
                else:
                    left = 5
                crop_margins = f"{left} 5 5 5"  # left top right bottom margins in pts
                subprocess.run(['pdfcrop', '--margins', crop_margins, output_path, output_path], check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                print(f"Figure saved to {output_path} (cropping failed - pdfcrop is not installed or not found in PATH)")
            print(f"Figure saved to {output_path}")
        else:
            print(f"File not saved. Skipping.")

    if show:
        plt.show()
    else:
        plt.close()

def linear_interpolate(x, z, y, f_x, f_z):
    if z == x: 
        return f_x
    interpolation_factor = (y - x) / (z - x)
    return f_x + (f_z - f_x) * interpolation_factor

def plot_and_save_games(logfolders, confidencepercentile=0.3, which_plots={'value-time': True, 'gap-iter': True, 'value-iter': False, 'gap-time': False}, max_length={'time': 3600, 'iter': 500}, algo_exclusions=[]):
    
    general_output_dir=os.path.join(library_dir, "plots/convergence")
    for folder in logfolders:
        if 'runs/' in folder:
            subfolder = folder.split('runs/', 1)[1]
            subfolder = os.path.dirname(subfolder)
            output_dir = os.path.join(general_output_dir, subfolder)
            os.makedirs(output_dir, exist_ok=True)
        else:
            print(f"Warning: 'runs/' not found in {folder}. Using the basename instead.")
            output_dir = general_output_dir
        
        plot_a_game(folder, show=False, output_dir=output_dir, confidencepercentile=confidencepercentile, which_plots=which_plots , max_length=max_length, algo_exclusions=algo_exclusions)

def plot_two(folder, show=True, output_dir='./plots/mainbody', confidencepercentile=0.3, add_value_iter=True, max_length = {'iter': 3000}, algo_exclusions = []):
    for i, logfolder in enumerate(folder):
        plot_game_best(logfolder, show=show, output_dir=output_dir, confidencepercentile=confidencepercentile, add_value_iter=add_value_iter, max_length=max_length, algo_exclusions=algo_exclusions, surpress = (i % 2))


def plot_and_save_best(logfolders, confidencepercentile=0.3, add_value_iter=True, max_length={'iter': 500}, algo_exclusions=[]):
    
    general_output_dir=os.path.join(library_dir, "plots/best_convergence")
    for folder in logfolders:
        # Extract the subfolder after the first occurrence of 'runs/'
        if 'runs/' in folder:
            subfolder = folder.split('runs/', 1)[1]
            subfolder = os.path.dirname(subfolder)
            output_dir = os.path.join(general_output_dir, subfolder)
            os.makedirs(output_dir, exist_ok=True)
        else:
            print(f"Warning: 'runs/' not found in {folder}. Using the basename instead.")
            output_dir = general_output_dir
        
        plot_game_best(folder, show=False, output_dir=output_dir, confidencepercentile=confidencepercentile, add_value_iter=add_value_iter, max_length = max_length, algo_exclusions = algo_exclusions)


if __name__ == "__main__":
    folder = "./experiments/game6communitydetection"

    plot_a_game(folder, which_plots = {'value-time': True, 'gap-iter': True, 'value-iter': False, 'gap-time': False}, max_length = {'time': 0.15, 'iter': 101})