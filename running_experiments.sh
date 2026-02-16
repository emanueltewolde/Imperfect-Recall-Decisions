#!/bin/bash

# Define parameter ranges

### BELOW ONLY SOLVES THE VERY SMALL DECISION PROBLEMS PREVIOUSLY GENERATED. UNCOMMENT THE SECOND LINE BELOW TO SOLVE THE FULL SIZE RANGE OF GENERATED DECISION PROBLEMS ###
games=( $(find "./decisionproblems/detection/a" -type f) $(find "./decisionproblems/random/a" -type f) $(find "./decisionproblems/simulation/a" -type f) )
# games=( $(find "./decisionproblems/detection/" -type f) $(find "./decisionproblems/random/" -type f) $(find "./decisionproblems/simulation/" -type f) )

algos=( "rm" "prm" "rm+" "prm+" "pgd_1e-3" "pgd_1e-2" "pgd_1e-1" "pgd_1e0" "optgd_1e-3" "optgd_1e-2" "optgd_1e-1" "optgd_1e0" "gurobi" )
for a in {-2..0}; do
    for b1 in 0.8 0.9 0.99; do
        for b2 in 0.99 0.999 0.9999; do
            algos+=("ams_a1e${a}_b${b1}_c${b2}")
        done
    done
done

### BELOW ONLY RUNS FOR THREE INITIALIZATIONS. FOR THE PAPER WE RUN THE SECOND LINE INSTEAD ###
seeds=( $(seq 33 1 35) )
# seeds=( $(seq 33 1 44) )

maxiters=( 6000 )
timelimits=14400
tols=( 1e-6 )

# Loop through all combinations
for game in "${games[@]}"; do
    for seed in "${seeds[@]}"; do
        for algo in "${algos[@]}"; do
            for maxiter in "${maxiters[@]}"; do
                for timelimit in "${timelimits[@]}"; do
                    for tol in "${tols[@]}"; do                        
                        # sbatch "./run_one_experiment.sh" "$game" "$algo" "$seed" "$maxiter" "$timelimit" "$tol"
                        bash "./run_one_experiment.sh" "$game" "$algo" "$seed" "$maxiter" "$timelimit" "$tol"
                    done
                done
            done
        done
    done
done
echo "All experiments started!"