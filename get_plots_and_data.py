import glob
from modules.evaluations.plot_convergence_per_game import plot_and_save_best, plot_and_save_games
from modules.evaluations.compare_times import summarize_data

folders_descr = "./runs/*/*/*"
logfolders = glob.glob(folders_descr)


summarize_data(logfolders, save=True)
plot_and_save_best(logfolders, confidencepercentile=0, add_value_iter=True, max_length={'iter': 3000}, algo_exclusions=[])


### BELOW PLOTS EACH GD (resp. OGD/AMS) PARAMETER SETTING AS AN INDIVIDUAL ALGORITHM ###

# # Build algo list
# algo_list = ["rm", "prm", "rm+", "prm+", "pgd_1e-3", "pgd_1e-2", "pgd_1e-1", "pgd_1e0", "optgd_1e-3", "optgd_1e-2", "optgd_1e-1", "optgd_1e0", "gurobi"]
# for a in [-2, -1, 0]:
#     for b1 in [0.8, 0.9, 0.99]:
#         for b2 in [0.99, 0.999, 0.9999]:
#             algo_list.append(f"ams_a1e{a}_b{b1}_c{b2}")

# algo_exclusions = [algo for algo in algo_list if not algo.startswith("pgd_")]
# plot_and_save_games(logfolders, confidencepercentile=0, which_plots={'value-time': True, 'gap-iter': True, 'value-iter': False, 'gap-time': False}, max_length={'time': 14400, 'iter': 3000}, algo_exclusions=algo_exclusions)
