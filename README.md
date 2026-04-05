Artificial Intelligence Project: HEX — Autonomous Hex Player 🧠♟️🔷🚀🧪

“Those who are crazy enough to think they can change the world are the ones who do.” — Steve Jobs 💡

Author: Jery Rodríguez Fernández — Group: C312 👤

Overview
--------
This repository implements an autonomous HEX player that combines classical adversarial search (Minimax with alpha–beta pruning) for small/medium boards and Monte Carlo Tree Search (MCTS) for larger boards. It also includes an automated parameter search module based on a Genetic Algorithm (GA) to optimize the evaluation heuristic. The design focuses on modularity, performance, and reproducible experiments. ⚙️📈🤖

Repository structure (technical summary) 🗂️
-------------------------------------
- `solution.py` — Experiment orchestrator and entrypoint. Responsibilities:
  - Read and expose configuration (board sizes, RNG seed, heuristic weight vectors, runtime options). ⚙️
  - Instantiate game engine and strategy implementations (`Minimax` or `MCTS`) according to board size and game phase. 🧩
  - Run matches and tournaments, log results, persist statistics and learned weight vectors. 📊
  - Load and apply weight combinations produced by `genetic_algorithm.py` for automated evaluation. 🔁

- `strategy.pdy` — Strategy and heuristic implementations used by Minimax and by MCTS rollouts. Key technical components:
  - Minimax with alpha–beta pruning and adaptive depth control (depth depends on board size and proximity to terminal nodes). ♟️✂️
  - Multi-signal heuristic combining: territorial control, minimum connection distance (BFS), connected-component metrics (count and largest component size), threatened-cell analysis, and prioritized move ordering. 🔎🧠
  - Move generator optimized for pruning: adjacent-to-stone moves first, then other candidates; intra-group ordering by Manhattan distance to strategic anchors (center/edges), dynamically adapted by game phase. 📐➡️
  - Heuristic parameters: 6-dimensional weight vector `(p1..p6)` that linearly combines normalized signals; `p6` is used as a phase threshold to switch spatial focus (center → edges). Implementations reuse intermediate graph representations (DSU, cached components) to speed evaluations and reduce redundant work. 🧩💾

- `genetic_algorithm.py` — Automated search of effective heuristic weight vectors using a Genetic Algorithm (GA). Operational details:
  - Representation: each individual is a 6D real-valued vector `(p1,...,p6)` encoding heuristic coefficients and the phase threshold. 🧬
  - Fitness: each individual plays `m` matches against a reference opponent (or opponent pool); fitness = win percentage (higher is better). 🏁
  - Evolutionary operators: selection (top-k or tournament), crossover with probability `p_c`, mutation with probability `p_m`. Hyperparameters: population `n`, matches per individual `m`, selection size `k`, crossover `p_c`, mutation `p_m`, generations `w`. 🔁⚙️
  - Outputs: persisted best vectors per generation; example strong solution found: `(p1..p6) = (16.92, 2.81, 5.15, 16.76, 1, 43.41)`. 🏆
  - Practical opening strategy: optionally sample randomly from a pre-seeded set of vectors to diversify early-game behavior. 🎲

Heuristic formula (compact) 🔢
----------------------------
The evaluation is a linear combination of normalized feature signals. In compact form:

$$
h\\big((i_1,\\dots,i_5),(p_1,\\dots,p_5)\\big) = p_1 i_1 + p_2 i_2 + p_3 i_3 + p_4 i_4 + p_5 i_5
$$

An additional scalar parameter $p_6$ controls a phase-based spatial focus (center vs edge) depending on the fraction of moves played. ⚖️

Algorithms & architecture (technical notes) 🧩
-----------------------------------------
- Minimax: alpha–beta pruning, iterative deepening (where applicable), transposition/caching table support, and heuristic move ordering to reduce branching factor. Time and memory trade-offs are configurable per board size. ⏱️💾
- MCTS: UCT-based selection with domain-specific rollout policy. Rollouts are biased towards adjacency to existing stones to produce higher-quality playouts for Hex. Tree and rollout layers use fast DSU/graph checks to detect wins early. 🌲🔍
- DSU (Disjoint Set Union) is heavily used for quick connection/win detection during both search and rollouts — amortized near-constant checks speed up simulations considerably. 🔗

Genetic Algorithm (GA) — practical settings and recommendations 🔬
----------------------------------------------------------
- Representation: float vector of length 6; use bounded ranges and normalization for numerical stability. 🔢
- Fitness evaluation: use parallelized match workers when evaluating many individuals to scale with CPU cores. 🧵⚡
- Operators: prefer simulated binary crossover (SBX) or uniform crossover for continuous genes; Gaussian or adaptive mutations for exploration. 🔁
- Hyperparameter tuning: balance `n`, `m`, and `w` depending on compute budget — larger `m` yields better fitness estimates but increases compute cost linearly. 📐

Optimizations & performance considerations 🛠️
---------------------------------------
- Cache feature computations (component counts, BFS distances) and invalidate incrementally on local moves. 🗃️
- Use prioritized move generators and shallow heuristic evaluations during alpha–beta to prune aggressively. ✂️
- Parallelize GA evaluations and, where safe, MCTS rollouts. Use modest locking or per-thread RNG streams to ensure reproducible results. 🔀

Quickstart (example) ▶️
---------------------
Run a simple match or experiment via `solution.py`. Example:

```bash
python solution.py --board-size 11 --strategy mcts --seed 42
```

For GA experiments:

```bash
python genetic_algorithm.py --population 50 --matches 20 --generations 100 --crossover 0.8 --mutation 0.02
```

