#!/bin/bash

#SBATCH --partition=...
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --exclude=...
#SBATCH --output=./debugging/%j_output.out
#SBATCH --error=./debugging/%j_error.err

type="$1"
size="$2"
i="$3"
num_per_size="$4"

output_dir="./decisionproblems/${type}/${size}"

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "[$start_time]Starting the $i th / $num_per_size generation of a game of the type $type and in size $size"
echo "[$start_time]Checking that error file works" >&2

python -m "modules.game_generators.one_game" \
    --type "$type" \
    --size "$size" \
    --i "$i" \
    --num_per_size "$num_per_size" \
    --output_dir "$output_dir"

end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "[$end_time] Finished the $i th / $num_per_size generation of a $size game of type=$type"