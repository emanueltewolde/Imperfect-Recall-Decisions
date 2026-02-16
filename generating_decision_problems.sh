#!/bin/bash

# Define parameter ranges
types=( "simulation" "random" "detection" )

### BELOW ONLY GENERATES VERY SMALL DECISION PROBLEMS. UNCOMMENT THE SECOND OR THIRD LINE BELOW TO INSTEAD GENERATE DECISION PROBLEMS UP TO MEDIUM SIZE OR OF FULL SIZE RANGE ###
sizes=('a' )
# sizes=('a' 'b' 'c')
# sizes=('a' 'b' 'c' 'xs' 's' 'm' 'l' 'xl') # sizes in increasing order, from hundreds to millions of nodes

num_per_size=3 # number of games per size
iterations=( $(seq 1 $num_per_size) )

for type in "${types[@]}"; do
    for size in "${sizes[@]}"; do
        for i in "${iterations[@]}"; do                      
            # sbatch "./generate_one_decision_problem.sh" "$type" "$size" "$i" "$num_per_size"
            bash "./generate_one_decision_problem.sh" "$type" "$size" "$i" "$num_per_size"
        done
    done
done
echo "All game generations started!"