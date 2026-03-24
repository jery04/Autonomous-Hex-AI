import random
import argparse
import time
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from solution import Minimax, HexGraph
from board import HexBoard
import math

Vector = List[float]

# Opponent center vector: opponents will be sampled in a neighbourhood around this
OPPONENT_CENTER = [16.9283, 2.8124, 5.1506, 16.7625, 1, 43.4640]

RANGES = [
    (50.0,  350.0),   # a  → distancia (el rey absoluto, casi siempre el más alto)
    (5.0,   120.0),   # b  → comp_opp - comp_self (penaliza fragmentación)
    (5.0,   100.0),   # c  → max_comp_self - max_comp_opp (premia cadenas largas)
    (10.0,  250.0),   # d  → threats_opp - threats_self (amenazas suelen ser críticas)
    (0.0,   80.0),    # e  → dom_self - dom_opp (dominio territorial, suele pesar menos)
    (3.0, 65.0),      # f
]

def neighbor_vector(center: Vector, delta: float = 10.0) -> Vector:
    """Return a vector where each component is center +/- r, with r ~ U(0, delta).
    The sign (plus or minus) is chosen at random per component and values are
    clamped to the per-component ranges in `RANGES`."""
    out = []
    for i, c in enumerate(center):
        lo, hi = RANGES[i]
        r = random.uniform(0.0, delta)
        if random.random() < 0.5:
            v = c - r
        else:
            v = c + r
        # Clamp to valid per-component range
        v = max(lo, min(hi, v))
        out.append(v)
    return out

def random_vector() -> Vector:
    """Sample a random vector using per-component ranges for a..f."""
    return [random.uniform(lo, hi) for (lo, hi) in RANGES]

def set_weights(weights: Vector) -> None:
    Minimax.distance, Minimax.components, Minimax.max_component, Minimax.threats, Minimax.territory, Minimax.ctrl_board = weights

def play_match(size: int, w1: Vector, w2: Vector, mitad: bool = True) -> int:
    """Play one game and return only the winner id (1 or 2).

    Returns 0 for draws or aborted games.
    """
    board = HexBoard(size)
    g1 = HexGraph(size=size, player_id=1)
    g2 = HexGraph(size=size, player_id=2)
    turn = 0
    max_moves = size * size
    
    if turn == 0:
        r = random.randint(0, size - 1)
        c = random.randint(0, size - 1)
        g1.mark_node_at(r, c, 1)
        board.place_piece(r, c, 1)
    else:
        r = random.randint(0, size - 1)
        c = random.randint(0, size - 1)
        g2.mark_node_at(r, c, 2)
        board.place_piece(r, c, 2)

    for _ in range(max_moves):
        move_t0 = time.time()
        if turn == 0:
            set_weights(w1)
            move = Minimax.preminimax(g1, board)
            pid = 1
        else:
            set_weights(w2)
            move = Minimax.preminimax(g2, board)
            pid = 2

        if move is None:
            return None

        r, c = move
        board.place_piece(r, c, pid)

        if mitad and g1.move_counter >= math.floor(0.32*size*size):
            var  = Minimax.calculate_heuristic(g1, g1.free_cells)
            if var > 0:
                return 1
            elif var < 0: 
                return 2
            else: mitad = False
        
        elif not mitad and board.check_connection(pid):
            return pid

        turn = 1 - turn

    # draw
    return None

def fitness(individuo: Vector, n_games: int = 30, p1: float = 0.5, p2: float = 0.5, board_sizes: List[int] = [4, 5]) -> float:
    """Return win percentage x100 vs random opponents on the provided board sizes.

    The function signature accepts two unused float placeholders (kept for
    compatibility with calls that pass extra parameters) and a `board_sizes`
    list which will be cycled through when selecting the board size for each
    match.
    """
    wins = 0
    if not board_sizes:
        board_sizes = [4, 5]
    for game_idx in range(n_games):
        opponent = OPPONENT_CENTER
        size = board_sizes[game_idx % len(board_sizes)]

        # Randomize who starts to reduce first-move bias.
        if random.random() < 0.5:
            winner_id = play_match(size, individuo, opponent)
            if winner_id == 1:
                wins += 1
        else:
            winner_id = play_match(size, opponent, individuo)
            if winner_id == 2:
                wins += 1

    return wins / n_games * 100.0

def init_population(pop_size: int) -> List[List[float]]:
    return [neighbor_vector(OPPONENT_CENTER, 100) for _ in range(pop_size)]

def tournament_selection(pop: List[Vector], fitnesses: List[float], k: int = 3) -> Vector:
    selected = random.sample(range(len(pop)), k)
    best = max(selected, key=lambda i: fitnesses[i])
    return pop[best][:]

def crossover(p1: Vector, p2: Vector, pc: float) -> Tuple[Vector, Vector]:
    if random.random() > pc:
        return p1[:], p2[:]
    alpha = random.random()
    c1 = [alpha * x + (1 - alpha) * y for x, y in zip(p1, p2)]
    c2 = [(1 - alpha) * x + alpha * y for x, y in zip(p1, p2)]
    return c1, c2

def mutate(ind: Vector, pm: float, sigma: float = 6.0) -> None:
    for i in range(len(ind)):
        if random.random() < pm:
            ind[i] += random.gauss(0, sigma)
            lo, hi = RANGES[i]
            ind[i] = max(lo, min(hi, ind[i]))

def ga_optimize(
    pop_size: int = 160,
    generations: int = 20,
    pc: float = 0.75,
    pm: float = 0.18,
    seed: int = None,
    top_frac: float = 0.05,
    n_games: int = 24,
    board_sizes: List[int] = [5, 4],
):
    if seed is not None:
        random.seed(seed)

    pop = init_population(pop_size)

    best_overall: Vector = []
    best_fit_overall = -1.0

    for gen in range(1, generations + 1):
        gen_t0 = time.time()
        # Parallelize fitness evaluation across available CPU cores
        with Pool(processes=min(cpu_count(), len(pop))) as pool:
            # pass board_sizes into fitness so evaluation uses selected board sizes
            fitnesses = pool.starmap(fitness, [(ind, n_games, 0.5, 0.5, board_sizes) for ind in pop])

        ranked = sorted(range(len(pop)), key=lambda i: fitnesses[i], reverse=True)

        # Print population ordered from highest to lowest fitness
        for i in ranked:
            ind = pop[i]
            fit = fitnesses[i]
            print(f"{fit:6.2f}% -> {[round(x, 4) for x in ind]}")
        best_idx = ranked[0]
        best_fit = fitnesses[best_idx]
        best_ind = pop[best_idx][:]

        if best_fit > best_fit_overall:
            best_fit_overall = best_fit
            best_overall = best_ind[:]

        top_n = max(1, int(top_frac * pop_size))
        top_pop = [pop[i][:] for i in ranked[:top_n]]
        top_fit = [fitnesses[i] for i in ranked[:top_n]]

        new_pop: List[Vector] = [ind[:] for ind in top_pop]

        while len(new_pop) < pop_size:
            p1 = tournament_selection(top_pop, top_fit, k=min(3, len(top_pop)))
            p2 = tournament_selection(top_pop, top_fit, k=min(3, len(top_pop)))
            c1, c2 = crossover(p1, p2, pc)
            mutate(c1, pm)
            mutate(c2, pm)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = new_pop

        gen_elapsed = time.time() - gen_t0
        print(f"Generacion {gen} terminada en {gen_elapsed:.2f}s")

    # After loop, `best_ind` and `best_fit` hold the last generation best
    last_best_ind = best_ind[:]
    last_best_fit = best_fit

    return best_overall, best_fit_overall, last_best_ind, last_best_fit

def main():
    parser = argparse.ArgumentParser(description="GA tuner for Minimax weights (a..f) based on match win rate")
    parser.add_argument("--pop", type=int, default=60)
    parser.add_argument("--gen", type=int, default=14)
    parser.add_argument("--pc", type=float, default=0.8)
    parser.add_argument("--pm", type=float, default=0.05)
    parser.add_argument("--n_games", type=int, default=71)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sizes",
        type=str,
        default="5,4",
        help="Comma-separated list of board sizes to evaluate, e.g. 3",
    )
    args = parser.parse_args()

    t0 = time.time()
    print("Iniciando optimización GA...")
    # parse sizes argument into a list of ints
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    best_overall, best_fit, last_best, last_fit = ga_optimize(
        pop_size=args.pop,
        generations=args.gen,
        pc=args.pc,
        pm=args.pm,
        seed=args.seed,
        n_games=args.n_games,
        top_frac=0.18,
        board_sizes=sizes,
    )
    elapsed = time.time() - t0
    print(f"Optimización completada en {elapsed:.2f}s")
    if last_best:
        print(f"Mejor individuo última generación: {[round(x,4) for x in last_best]} -> {last_fit:.2f}%")
    if best_overall:
        print(f"Mejor individuo global: {[round(x,4) for x in best_overall]} -> {best_fit:.2f}%")
    else:
        print("No se encontró un mejor individuo global.")

if __name__ == "__main__":
    main()
