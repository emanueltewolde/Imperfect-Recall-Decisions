
import argparse
import os
import logging
import numpy as np
import functools

from modules.IRgame import IRgame
from modules.solvers.FirstOrderMethod import FirstOrderMethod
from modules.solvers.rm import RegretMatching
from modules.solvers.gd import ProjectedGradientDescent, OptimisticGradientDescent, AMSGrad
from modules.solvers.SolveWithGurobi import solve_with_gurobi
from modules.utils import logging_frequency


def main():
    parser = argparse.ArgumentParser(description='Run experiment with parameters')
    parser.add_argument('--gamefile', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, required=False, default=None)
    parser.add_argument('--maxiter', type=int, required=False, default=None)
    parser.add_argument('--timelimit', type=float, required=False, default=None)
    parser.add_argument('--tol', type=float, required=False, default=None)
    args = parser.parse_args()

    if args.algo == "gurobi":
        #go one folder up
        last_folder = os.path.basename(os.path.normpath(args.output_dir))
        if last_folder.isdigit():
            args.output_dir = os.path.dirname(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    game_basename = os.path.basename(args.gamefile)
    if game_basename.endswith('.txt'):
        game_basename = os.path.splitext(game_basename)[0] 
    
    if args.algo == "gurobi":
        log_file = os.path.join(args.output_dir, f"{game_basename}__algo-{args.algo}__timelimit-{args.timelimit}__tol-{args.tol}.log")
        if os.path.exists(log_file):
            # os.remove(log_file)
            log_file = os.path.join(args.output_dir, f"{game_basename}__algo-{args.algo}__timelimit-{args.timelimit}__tol-{args.tol}__blockedreruns.log")
            logging.basicConfig(
                filename=log_file,
                filemode='a',
                level=logging.INFO,
                format='%(asctime)s -- %(message)s'
            )
            logging.info(f"# Blocked a run of algorithm {args.algo} on game file {game_basename}. The other parameters are: Tolerance = {args.tol}, Time limit = {args.timelimit}, Max Iterations = {args.maxiter}, Random seed {args.seed}.")
            return
    else:
        log_file = os.path.join(args.output_dir, f"{game_basename}__algo-{args.algo}__seed-{args.seed}__maxiter-{args.maxiter}__timelimit-{args.timelimit}__tol-{args.tol}.log")

        version = 2
        while os.path.exists(log_file):
            log_file = os.path.join(args.output_dir, f"{game_basename}__algo-{args.algo}__seed-{args.seed}__maxiter-{args.maxiter}__timelimit-{args.timelimit}__tol-{args.tol}_v{version}.log")
            version += 1

    

    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s -- %(message)s'
        )
    # logging.info(f"# Run of algorithm {args.algo} on game file {game_basename}. The other parameters are: Tolerance = {args.tol}, Time limit = {args.timelimit}, Max Iterations = {args.maxiter}, Random seed {args.seed}.")
    logging.info(f"# Game file = {game_basename}")
    logging.info(f"# Game ID = {None}")
    logging.info(f"# Algorithm = {args.algo}")
    logging.info(f"# Tolerance = {args.tol}")
    logging.info(f"# Time limit = {args.timelimit}")
    logging.info(f"# Maximal number of iterations = {args.maxiter}")
    logging.info(f"# Random seed {args.seed}")
    logging.info("# ")

    run_experiment(args)

def run_experiment(args):
    logging.info("# Starting to build the game.")
    game = IRgame(args.gamefile)
    logging.info("# Finished building the game.")

    if args.algo == "gurobi":
        logging.info("# Starting to setup the solver and solve the game.")
        solver = solve_with_gurobi(game)


        result = solver.solve(timelimit=args.timelimit, tol=args.tol, output_settings=0)
        logging.info(f"# Gurobi optimizer status: {result['status']}")
        logging.info(f"# Final value: {result['value']}")
        logging.info(f"# Time needed to solve: {result['time']}")
        logging.info(f"# Final point: {result['x']}")
        logging.info("# Experiment run completed")
    else:
        logging.info("# Starting to generate starting point.")
        #random number generator for repeatably calling a random function
        np_rng = np.random.RandomState(args.seed)
        start = game.get_random_point(method = "exponential", np_rng=np_rng)
        logging.info("# Finished generating starting point.")
        logging.info(f"# Starting point: {start}")

        logging.info("# Starting to setup the solver and solve the game.")
        setup_and_solve(args.algo, game, start, args.maxiter, args.timelimit, args.tol)    
        logging.info("# Experiment run completed")
    
def setup_and_solve(algo_str, game, startpoint, maxiter, timelimit, tol):
        if algo_str == "rm":
            local_opt = functools.partial(RegretMatching, plus=False, predictive=False)
        elif algo_str == "prm":
            local_opt = functools.partial(RegretMatching, plus=False, predictive=True)
        elif algo_str == "rm+":
            local_opt = functools.partial(RegretMatching, plus=True, predictive=False)
        elif algo_str == "prm+":
            local_opt = functools.partial(RegretMatching, plus=True, predictive=True)
        elif algo_str[:3] == "pgd":
            assert algo_str[3] == "_" and algo_str[5] == "e"
            step_size = int(algo_str[4]) * 10**int(algo_str[6:])
            logging.info(f"# Step size of PGD: {step_size}")
            local_opt = functools.partial(ProjectedGradientDescent, lr={'mode': 'constant', 'init_rate': step_size})
        elif algo_str[:5] == "optgd":
            assert algo_str[5] == "_" and algo_str[7] == "e"
            step_size = int(algo_str[6]) * 10**int(algo_str[8:])
            logging.info(f"# Step size of Optimistic GD: {step_size}")
            local_opt = functools.partial(OptimisticGradientDescent, lr={'mode': 'constant', 'init_rate': step_size})
        elif algo_str[:3] == "ams":
            assert algo_str[3:5] == "_a"
            a, rest = algo_str[5:].split("_b")
            assert "_c" in rest
            b1, b2 = rest.split("_c")
            beta1 = float(b1)
            beta2 = float(b2)
            alpha = float(a)
            logging.info(f"# alpha, beta1, and beta2 in AMSGrad are: {alpha}, {beta1}, {beta2}")
            local_opt = functools.partial(AMSGrad, params={'mode': 'constant', 'init_rate': alpha, 'beta1': beta1, 'beta2': beta2}, deal_with_zeros={'method': 'limit'})
        else:
            raise ValueError(f"Unknown algorithm: {algo_str} with the type {type(algo_str)}")
        
        solver = FirstOrderMethod(game, local_opt)
        result = solver.solve(init=startpoint, max_iter=maxiter, timelimit=timelimit, tol=tol, reporting={'report_style': "logging", 'report_freq': logging_frequency(time_not_iter=False)})
        logging.info(f"# Time needed to solve: {result['time']}")
        logging.info(f"# Final point: {result['x']}")

    
if __name__ == "__main__":
    main()