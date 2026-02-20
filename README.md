# Decision Making under Imperfect Recall: Algorithms and Benchmarks

This repository contains the research framework on decision problems with imperfect recall described in the associated paper:

"**[Decision Making under Imperfect Recall: Algorithms and Benchmarks](https://emanueltewolde.com/files/IRPractical.pdf)**" by Emanuel Tewolde, Brian Hu Zhang, Ioannis Anagnostides, Tuomas Sandholm, and Vincent Conitzer.

It provides a complete pipeline for:

1. **Generating** decision problems with imperfect recall across various domains
2. **Solving** them using state-of-the-art optimization algorithms
3. **Evaluating** and comparing algorithm performance through comprehensive metrics and visualizations

The repository implements 8 (families of) algorithms across three benchmark problem types, ranging from hundreds to millions of nodes.

**Citation:**

```bibTeX
@article{Tewolde2026:Decision,
    author = {Emanuel Tewolde and Brian Hu Zhang and Ioannis Anagnostides and Tuomas Sandholm and Vincent Conitzer},
    title = {Decision Making under Imperfect Recall: Algorithms and Benchmarks},
    year = {2026},
    journal={arXiv:2602.15252}
}
```

---

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Downloading the Full Benchmark Suite (Optional)](#downloading-the-full-benchmark-suite-optional)
- [Benchmark Problem Types](#benchmark-problem-types)
- [Algorithms Implemented](#algorithms-implemented)
- [Additional Notes](#additional-notes)
- [File Format Specification](#file-format-specification)
- [Configuration and Customization](#configuration-and-customization)
- [Output and Visualization](#output-and-visualization)

---

## Project Structure

```
.
├── modules/                              # Core Python implementation
│   ├── game_generators/                  # Benchmark generation
│   │   ├── one_game.py                   # Main generation entry point
│   │   ├── random.py                     # Random decision trees
│   │   ├── simulation.py                 # AI safety testing scenarios
│   │   ├── detection.py                  # Subgroup detection problems
│   │   ├── graph_generators.py           # Graph generation utilities
│   │   └── writegamefile.py              # File output formatter
│   ├── solvers/                          # Optimization algorithms
│   │   ├── FirstOrderMethod.py           # Base solver class
│   │   ├── rm.py                         # Regret Matching methods (4 variants)
│   │   ├── gd.py                         # Gradient methods (3 families)
│   │   └── SolveWithGurobi.py            # Commercial solver interface
│   ├── evaluations/                      # Experiment execution & analysis
│   │   ├── one_run.py                    # Single experiment runner
│   │   ├── compare_times.py              # Results aggregation
│   │   └── plot_convergence_per_game.py  # Visualization
│   ├── preprocessing/                    # File validation & parsing
│   │   ├── check_format.py               # Format validator
│   │   └── full_info_game.py             # Game tree constructor
│   ├── IRgame.py                         # Main game object class
│   ├── tree.py                           # Node and InfoSet data structures
│   └── utils.py                          # Utility functions
├── decisionproblems/                     # Generated benchmark problems
│   ├── generation_instructions/          # YAML configuration files
│   │   ├── randomstandard.yaml
│   │   ├── simulationstandard.yaml
│   │   ├── detectionstandard.yaml
│   │   └── Implemented.txt               # Configuration documentation
│   └── simple/                           # Small test instances for debugging
├── plots/                                # Output visualizations
│   ├── best_convergence/                 # Best algorithm configurations
│   ├── convergence/                      # All configurations
│   └── values_and_times_summary/         # CSV results
├── runs/                                 # Algorithm execution logs
├── generating_decision_problems.sh       # Batch generation script
├── generate_one_decision_problem.sh      # Single generation script
├── running_experiments.sh                # Batch experiment script
├── run_one_experiment.sh                 # Single experiment script
├── get_plots_and_data.py                 # Main analysis script
└── environment.yml                       # Conda environment specification
```

---

## Getting Started

### Prerequisites

- **Conda** (Anaconda or Miniconda)
- **Gurobi Optimizer** with a valid license
  - Academic licenses are free and available at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/)
  - The license must be activated on your system before running experiments

### Installation

1. **Clone or download this repository:**
   ```bash
   cd /path/to/github_repo
   ```

2. **Create and activate the Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate IRpractical
   ```
   This installs all required dependencies.

### Quick Start

Run the complete pipeline on small test instances:

```bash
# 1. Generate small decision problems (takes ~1 minute)
bash generating_decision_problems.sh

# 2. Run algorithms on generated problems (takes ~1 hour. If that is too long for you, you could remove AMSGrad and other algorithms from being tested)
bash running_experiments.sh

# 3. Analyze results and create plots (takes ~2 minutes)
python get_plots_and_data.py
```

**Expected Output:**
- Generated games: `./decisionproblems/{problem_type}/a/*.txt`
- Experiment logs: `./runs/{problem_type}/a/{game_id}/{seed_num}/*.log` and `./runs/{problem_type}/a/{game_id}/{gurobi_run}.log`
- Summary CSV: `./plots/values_and_times_summary/results1.csv`
- Convergence plots: `./plots/best_convergence/*.pdf`

### Downloading the Full Benchmark Suite (Optional)
The full benchmark suite used in the research paper, comprising 61 decision problems, is too large for repository inclusion and therefore provided via [this Google Drive Link](https://drive.google.com/file/d/1wdE2Em-CtRYiE0PFjvJuQpSyZGalE6ZS/view?usp=sharing).

```bash
# Install gdown and download the zipped folder from google drive
pip install gdown
gdown 1wdE2Em-CtRYiE0PFjvJuQpSyZGalE6ZS

# Unzip and move the decision problems to their respective folder. This first removes all files currently under `./decisionproblems/{simulation,random,detection}`, and cleans up temporary files at the end
rm -rf ./decisionproblems/{simulation,random,detection}
unzip IRbenchmark_full.zip -d ./temp_benchmark
mv ./temp_benchmark/decisionproblems/{simulation,random,detection} ./decisionproblems/
rm -rf ./temp_benchmark
rm IRbenchmark_full.zip
```

---

## Benchmark Problem Types

The framework generates three types of decision problems, each modeling different real-world scenarios:

### 1. AI Safety Testing Scenarios (`simulation`)

Models an AI agent that must choose between good and bad actions across multiple test scenarios before real-world deployment.

**Use case:** Evaluating AI systems under uncertainty with limited testing budget before deployment.

**Structure:**
- Multiple test scenarios for evaluation
- Agent can be tested a limited number of times (stochastically decided)
- Must then act in real deployment scenarios
- Good actions in real deployment yield positive utility; bad actions in real deployment yield extra positive utility, and in simulation cause the game to end with zero utility
- Imperfect recall: agent cannot distinguish between being in a test or deployment, or what scenarios it has encountered how often

### 2. Subgroup Detection under Privacy Constraints (`detection`)

A decision-maker selects nodes in a graph to maximize coverage of valuable hidden subgroups (communities) while respecting privacy constraints that limit information retention across selection rounds.

**Use case:** Pattern detection under privacy-preserving constraints.

**Structure:**
- Underlying graph structure (grid, Erdős-Rényi G(n,p), or G(n,m))
- Hidden positive nodes forming valuable subgroups (stars, lines, cliques, etc.)
- Decision-maker can inquire about nodes over multiple rounds
- Reward based on valuations of discovered subgroup nodes 
- Imperfect recall: only positive inquiries (nodes belonging to subgroups) are remembered; non-positive inquiries are forgotten

### 3. Random Decision Trees (`random`)

General tree-form decision problems with configurable imperfect recall structure, providing diverse test cases for algorithm benchmarking.

**Use case:** Basic benchmark for testing algorithm scalability, robustness, and performance across varied problem structures.

**Configurable parameters**:
- Tree depth bounds (minimum and maximum depth)
- Chance node presence probability
- Branching factors for chance and decision nodes
- Information set sizes (as proportion of total nodes)
- Payoff distributions

### Size Categories

Problems can be generated in 8 size categories:

| Size | Description | Approximate # of Nodes |
|------|-------------|------------------------|
| `a`  | Very Tiny   | Hundreds               |
| `b`  | Quite Tiny  |                        |
| `c`  | Tiny        |                        |
| `xs` | Extra Small |                        |
| `s`  | Small       |                        |
| `m`  | Medium      | Up to millions         |
| `l`  | Large       |                        |
| `xl` | Extra Large | (Multi-)million        |

**Default:** Scripts generate size `a` only. Uncomment lines in `generating_decision_problems.sh` to generate larger instances.

---

## Algorithms Implemented

The framework implements **8 families of algorithms**:

### 1. Regret Matching Methods (4 algorithms)

Classic algorithms for adaptive game-theoretic optimization:

- **RM:** Standard Regret Matching
- **RM+:** Regret Matching Plus
- **PRM:** Predictive Regret Matching
- **PRM+:** Predictive Regret Matching Plus

**Key feature:** No learning rate tuning required.

### 2. Projected Gradient Descent (4 parameter configurations)

Standard gradient descent with projection onto probability simplices:

- **PGD:** 4 learning rates: `1e-3`, `1e-2`, `1e-1`, `1e0`

**Key feature:** Simple, interpretable baseline.

### 3. Optimistic Gradient Descent (4 parameter configurations)

Gradient descent with optimistic (predictive) updates and projection onto probability simplices:

- **OGD:** 4 learning rates: `1e-3`, `1e-2`, `1e-1`, `1e0`

**Key feature:** Faster convergence in zero-sum games and minimax optimization.

### 4. AMSGrad (27 parameter configurations)

Adaptive learning rate and projection method (variant of Adam):

- **AMSGrad:** Grid search over:
  - Learning rate a ∈ {`1e-2`, `1e-1`, `1e0`}
  - Momentum b ∈ {`0.8`, `0.9`, `0.99`}
  - Second moment c ∈ {`0.99`, `0.999`, `0.9999`}
  - Total: 3 × 3 × 3 = 27 configurations

**Key feature:** Adaptive rescaling of gradients for momentum-based learning rates and projection onto probability simplices.

### 5. Commercial Solver (1 variant)

- **Gurobi:** Nonlinear optimization solver (requires license)

**Key feature:** Baseline for global optimality.

### Algorithm Naming Convention

In scripts and logs, algorithms are specified as:
- Regret Matching: `rm`, `rm+`, `prm`, `prm+`
- Gradient methods: `pgd_1e-3`, `optgd_1e-1`, etc.
- AMSGrad: `ams_a1e-2_b0.9_c0.999`
- Commercial: `gurobi`

---

## Additional Notes

### Slurm

Use `sbatch` instead of `bash` in `./generating_decision_problems.sh` and `./running_experiments.sh` for SLURM cluster parallelization.

### Experiment Parameters

**Default:**
- **maxiter:** 6000 iterations
- **timelimit:** 14400 seconds (4 hours), and 43200 for detection/xl problems
- **tol:** 1e-6 (KKT gap tolerance for convergence)
- **seeds:** 33-44 (full paper experiments)

### Comparing Algorithms

```bash
python get_plots_and_data.py
```
Plots the performance of all the algorithms with confidence intervals. (For PGD, OGD, and AMSGrad, the best variant is considered)

**Optional:** Plot all algorithm configurations (not just best):
```python
# Uncomment lines 15-23 in get_plots_and_data.py
plot_and_save_games(logfolders, ...)
```

This creates plots in `./plots/convergence/` showing all 40 algorithm variants.

---

## File Format Specification

Decision problems are stored in the text format used in LiteEFG [Liu et al., 2024]. Example (`absentminded_driver`):

```
# Comments start with #

# Decision node: player 1 chooses 'e' (exit) or 'c' (continue)
node / player 1 actions e c

# Leaf node: if exit immediately, payoff is 0
node /P1:e leaf payoffs 1=0

# Another decision node: same choice after continuing
node /P1:c player 1 actions e c

# Leaf nodes
node /P1:c/P1:e leaf payoffs 1=4
node /P1:c/P1:c leaf payoffs 1=0

# Information set: player 1 cannot distinguish between root and second decision
infoset pl1_1__highway/ nodes / /P1:c
```

### Format Rules

1. **Node definitions:**
   - Decision node: `node <path> player <player_id> actions <action1> <action2> ...`
   - Chance node: `node <path> chance actions <action1>=<prob1> <action2>=<prob2> ...`
   - Leaf node: `node <path> leaf payoffs <player_id>=<payoff>`

2. **Path notation:** `/Parent:action/Child:action/...`
   - Example: `/P1:left/C:heads/P1:up` means "player 1 chose left, chance chose heads, player 1 chose up"

3. **Information sets:**
   - `infoset <infoset_name> nodes <node_path1> <node_path2> ...`
   - All nodes in an infoset must have the same available actions

4. **Validation:** Run `python -m modules.preprocessing.check_format <filepath>` to take various types of checks on the format.

---

## Configuration and Customization

### Modifying Problem Generation

Edit YAML files in `./decisionproblems/generation_instructions/`:

**Example: Change AI safety testing budget**

Edit `simulationstandard.yaml`:
```yaml
testing:
  method: "chance"
  reroll_prob: 0.5  # was 0.7
  num_tests_max: 5  # was 4
```

See `Implemented.txt` for a description of all configurable parameters.

### Adding New Algorithms

1. **Implement solver** in `./modules/solvers/`:
   - Define it to work with the `FirstOrderMethod` class
   - Implement `__init__` and local optimizer

2. **Register algorithm** in `running_experiments.sh`:
   ```bash
   algos+=("myalgo_param1_param2")
   ```

3. **Parse algorithm string** in `modules/evaluations/one_run.py` (if needed)

### Customizing Experiments

Edit `running_experiments.sh`. Examples:

```bash
# Run only specific algorithms
algos=( "rm+" "pgd_1e-2" "optgd_1e-2" )

# More random seeds for statistical significance
seeds=( $(seq 1 1 50) )

# Longer time limit for large instances
timelimits=86400  # 24 hours
```

---

## Output and Visualization

### Log File Structure

Each experiment produces a timestamped log file.

```
2026-02-07 15:47:04,481 -- # Game file = agame2
2026-02-07 15:47:04,482 -- # Algorithm = ams_a1e-1_b0.99_c0.99
2026-02-07 15:47:04,482 -- # Tolerance = 1e-06
2026-02-07 15:47:04,482 -- # Time limit = 14400.0
2026-02-07 15:47:04,482 -- # Maximal number of iterations = 6000
2026-02-07 15:47:04,482 -- # Random seed 34
...
### Logging files such as below for first-order methods, or the solver output for Gurobi
2026-02-07 15:47:04,484 -- iter 0 | time 0.001 | value 2.125 | gap 3.509
2026-02-07 15:47:04,492 -- iter 10 | time 0.010 | value 4.273 | gap 0.801
...
2026-02-07 15:47:04,547 -- iter 78 | time 0.064 | value 6.0 | gap 1.11e-16
### End of Logging files, followed by solution summary
2026-02-07 15:47:04,547 -- # Time needed to solve: 0.064
2026-02-07 15:47:04,548 -- # Final point: [0.145, 0.490, 0.365, ...]
2026-02-07 15:47:04,548 -- # Experiment run completed
```

**Key metrics:**
- **iter:** Iteration number (first-order methods only)
- **time:** Elapsed wall-clock time in seconds
- **value:** Objective value (player utility)
- **gap:** KKT gap (stationarity measure; 0 = optimal)

### Summary CSV

`./plots/values_and_times_summary/results1.csv` contains:

| Column | Description |
|--------|-------------|
| `game` | Game identifier |
| `algo` | Algorithm name |
| `time_limit` | Time limit in seconds |
| `maxiter` | Maximum iterations (None for Gurobi) |
| `tol` | KKT gap tolerance |
| `found` | Whether a solution was found |
| `iter_limit_reached` | Whether iteration limit was hit |
| `time_limit_reached` | Whether time limit was hit |
| `value_median` | Median objective value across seeds |
| `time_needed_median` | Median solve time across seeds (seconds) |
| `gap` | Median final KKT gap (inf for Gurobi) |

### Plots

**Best convergence plots** (`./plots/best_convergence/`):
- For each game and seed, select the best configuration from PGD, OGD, and AMSGrad families
- Shows mean across seeds with confidence intervals

**All configurations plots** (`./plots/convergence/`):
- Shows all 40 algorithm variants

**Plot types:**
- **value-time:** Objective value vs. wall-clock time
- **gap-iter:** KKT gap vs. iteration (log scale KKT gap)
- **value-iter:** Objective value vs. iteration
- **gap-time:** KKT gap vs. wall-clock time


---

## Thank you

...for your interest in our work! Feel free to post a github issue if you have any questions.
