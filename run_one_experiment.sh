#!/bin/bash

#SBATCH --partition=...
#SBATCH --mem=16G
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=1
#SBATCH --exclude=...
#SBATCH --output=./debugging/%j_output.out
#SBATCH --error=./debugging/%j_error.err

game="$1"
algo="$2"
seed="$3"
maxiter="$4"
timelimit="$5"
tol="$6"

echo "gamepath: $game"
game_dir="$(dirname "$game")"
game_dir_after=$(echo "$game_dir" | grep -o "decisionproblems/.*" | sed 's|decisionproblems/||')
echo "game_dir_after: $game_dir_after"
game_basename="${game##*/}"
game_basename="${game_basename%.txt}"
output_dir="./runs/${game_dir_after}/${game_basename}/${seed}"

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "[$start_time] Starting experiment with game=$game_basename, algo=$algo, seed=$seed, maxiter=$maxiter, timelimit=$timelimit, tol=$tol"
echo "[$start_time] Checking that error file works" >&2

python -m "modules.evaluations.one_run" \
    --game "$game" \
    --algo "$algo" \
    --seed "$seed" \
    --output_dir "$output_dir" \
    --maxiter "$maxiter" \
    --timelimit "$timelimit" \
    --tol "$tol"

end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "[$end_time] Finished experiment with game=$game_basename, algo=$algo, seed=$seed, maxiter=$maxiter, timelimit=$timelimit, tol=$tol"