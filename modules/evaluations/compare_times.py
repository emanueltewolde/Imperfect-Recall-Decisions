# Creates a big csv file with all the values and times of the runs, and additional information

import re
import os
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

script_dir = os.path.dirname(os.path.abspath(__file__))
library_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

pattern = r'iter (\d+) \| time ([\d.e-]+) \| value ([\d.e-]+) \| gap ([\d.e-]+)'

def extract_from_my_logging_format(result, logfile, gurobi, time_limit, maxiter):
    with open(logfile, 'r') as file:
        log_text = file.read()

    lines = log_text.strip().split('\n')
    
    found = False

    for i, line in enumerate(lines):
        # Check if the line has a double dash
        if not ' --' in line:
            continue
        parts = line.split(' --', 1)
        content = parts[1]
        
        if gurobi and content.startswith(' # Gurobi optimizer status:'):
            assert ' # Experiment run completed' in lines[i + 4]
            iter = None
            value = float(lines[i + 1].split('# Final value: ', 1)[1].strip())
            time = float(lines[i + 2].split('# Time needed to solve: ', 1)[1].strip())
            time_reached = (time >= time_limit)
            if time_reached:
                gap = lines[i - 1].split(', gap ', 1)[1].strip()  
            else:
                gap = None
            iter_reached = False
            found = True
            break
        elif not gurobi and content.startswith(' # Time needed to solve: '):
            assert ' # Experiment run completed' in lines[i + 2]
            final = lines[i - 1].split(' --', 1)[1].strip()
            match = re.search(pattern, final)
            if match:
                iter = int(match.group(1))
                time = float(match.group(2))
                value = float(match.group(3))
                gap = float(match.group(4))
                time_reached = (time >= time_limit)
                iter_reached = (iter >= maxiter-1)
                found = True                
            else:
                print(f"Warning: No fitting final line of solving in {logfile}")
            break       

    if not found:
        iter = None
        value = None
        time = time_limit
        time_reached = True
        iter_reached = True
        gap = None      
        if not gurobi: print(f"Warning: No valid results found in {logfile}.")

    result.update({
        'found': found,
        'iter_needed': iter,
        'iter_reached': iter_reached,
        'time_needed': time,
        'time_reached': time_reached,
        'value': value,
        'gap': gap
    })
    return result

def get_values_and_times_raw(logfolders, save=True):

    data = []
    for log_folder in logfolders:
        # Extract the subfolder after the first occurrence of 'runs/'
        if 'runs/' in log_folder:
            game_descr = log_folder.split('runs/', 1)[1]
            # game_descr = os.path.dirname(subfolder)
        else:
            raise Exception(f"Warning: 'runs/' not found in {log_folder}. Using the basename instead.")
        
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

                algo = info[1].replace('algo-', '')
                result = {}

                if algo == 'gurobi':
                    if info[-1].startswith('blockedreruns'):
                        continue
                    
                    time_limit = float(info[2].replace('timelimit-', ''))
                    tol = float(info[3].replace('tol-', '').replace('.log', ''))
                    result['game'] = game_descr
                    result['algo'] = algo
                    result['seed'] = "None"
                    result['time_limit'] = time_limit
                    result['maxiter'] = "None"
                    result['tol'] = tol
                    result = extract_from_my_logging_format(result, full_path, gurobi=True, time_limit=time_limit, maxiter=None)
                else:
                    seed = int(info[2].replace('seed-', ''))
                    maxiter = int(info[3].replace('maxiter-', ''))
                    time_limit = float(info[4].replace('timelimit-', ''))
                    tol = float(info[5].replace('tol-', '').replace('.log', ''))
                    result['game'] = game_descr
                    result['algo'] = algo
                    result['seed'] = seed
                    result['time_limit'] = time_limit
                    result['maxiter'] = maxiter
                    result['tol'] = tol
                    result = extract_from_my_logging_format(result, full_path, gurobi=False, time_limit=time_limit, maxiter=maxiter)
                
                data.append(result)
                
        assert name_updates == 1
    
    data = pd.DataFrame(data)  
    
    if save:
        output_dir=os.path.join(library_dir, "plots/values_and_times_raw/")
        saving_number = next_file_number(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'results{saving_number}')
        data.to_csv(output_file + '.csv', index=False)
        print(f"Raw data saved to {output_file}.csv")
    
    return data

def summarize_data(folder, save=True):
    data = get_values_and_times_raw(folder, save=False)

    # Process each game-algo pair separately and build the summary dataframe
    game_algo_pairs = data[['game', 'algo']].drop_duplicates().values
    summary_data = []
    
    for game, algo in game_algo_pairs:
        subset = data[(data['game'] == game) & (data['algo'] == algo)]
        
        # Check if all entries have the same values for time_limit, maxiter, and tol
        time_limits = subset['time_limit'].unique()
        maxiters = subset['maxiter'].unique()
        tols = subset['tol'].unique()

        if len(time_limits) > 1:
            print(f"Warning: Multiple time_limit values for {game}, {algo}: {time_limits}")
        if len(maxiters) > 1:
            print(f"Warning: Multiple maxiter values for {game}, {algo}: {maxiters}")
        if len(tols) > 1:
            print(f"Warning: Multiple tol values for {game}, {algo}: {tols}")
        
        # Print out entries where 'found' is False
        not_found_entries = subset[subset['found'] == False]
        if not not_found_entries.empty:
            found = False
            if algo != 'gurobi':
                print(f"\nWarning: Found entries with 'found'=False for {game}, {algo}:")
                print(not_found_entries)
        else:
            found = True
        
        # Check if any runs hit the iteration limit
        iter_limit_reached = any(subset['iter_reached'] == True)
        time_limit_reached = any(subset['time_reached'] == True)

        gap_values = subset['gap'].copy()
        gap_values = gap_values.fillna(float('inf'))
        # No need for infer_objects as we're already handling the types properly
        # Convert string percentages to float values
        if isinstance(gap_values.iloc[0], str):
            gap = gap_values.iloc[0]
            assert len(gap_values) == 1
            assert '%' in gap or '-' in gap
        else:
            gap = gap_values.median()

        # Replace NaN gaps with 0 for median calculation
        value_values = subset['value'].copy()
        value_values = value_values.fillna(float('-inf'))
        value_median = value_values.median()

        # if subset['value'].isna().any():
        #     value_mean = float('nan')
        #     value_std = float('nan')
        # else:
        #     value_mean = subset['value'].mean()
        #     # Calculate standard deviation
        #     value_std = 0.0 if len(subset) == 1 else subset['value'].std()
                
        summary_row = {
            'game': game,
            'algo': algo,
            'time_limit': time_limits[0],
            'maxiter': maxiters[0],
            'tol': tols[0],
            'found': found,
            'iter_limit_reached': iter_limit_reached,
            'time_limit_reached': time_limit_reached,
            'value_median': value_median,
            # 'value_std': value_std,
            'time_needed_median': subset['time_needed'].median(),
            # 'time_needed_std': 0.0 if len(subset) == 1 else subset['time_needed'].std(),
            'gap': gap
        }
        
        summary_data.append(summary_row)
    
    # Create the summary DataFrame
    summary = pd.DataFrame(summary_data)
    
    # Flatten the multi-level column index
    # No need to flatten columns since they're already flat in this case
    # Just ensure column names are properly formatted
    if isinstance(summary.columns, pd.MultiIndex):
        summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in summary.columns]
    
    if save:
        # Save the summary data
        output_dir = os.path.join(library_dir, "plots/values_and_times_summary/")
        os.makedirs(output_dir, exist_ok=True)
        saving_number = next_file_number(output_dir)
        output_file = os.path.join(output_dir, f'results{saving_number}')
        summary.to_csv(output_file + '.csv', index=False)
        print(f"Summary data saved to {output_file}.csv")
    
    return summary

def next_file_number(folder_path):
    files = os.listdir(folder_path)
    pattern = re.compile(r'^results(\d+)')
    max_number = 0
    for filename in files:
        match = pattern.match(filename)
        if match:
            if int(match.group(1)) > max_number:
                max_number = int(match.group(1))
    
    if max_number >=1:
        return max_number + 1
    else:
        return 1


if __name__ == "__main__":
    logfolder = ["./runs/detection/xs/xsgame2"]

    # hi = get_values_and_times_raw(folder, save=False)
    hi = summarize_data(logfolder, save=False)
    print(hi)