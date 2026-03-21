import random
import argparse
import time
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from solution import Minimax, HexNodeGraph
from board import HexBoard

Vector = List[float]

# Opponent center vector: opponents will be sampled in a neighbourhood around this
OPPONENT_CENTER = [120.0, 10.0, 18.0, 60.0, 35.0, 28.0]
RANGES = [
    (80.0, 220.0),  # a
    (4.0, 35.0),    # b
    (8.0, 50.0),    # c
    (30.0, 140.0),  # d
    (15.0, 90.0),   # e
    (18.0, 42.0),   # f
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
    Minimax.a, Minimax.b, Minimax.c, Minimax.d, Minimax.e, Minimax.f = weights

def play_match(size: int, w1: Vector, w2: Vector) -> tuple:
    """Play one game and return (winner id, winner_move_count, avg_seconds_per_move)

    Time is tracked per player and the function returns the average seconds
    per move for the winner. For draws or aborted games returns (0,0,0).
    """
    board = HexBoard(size)
    g1 = HexNodeGraph(size=size, player_id=1)
    g2 = HexNodeGraph(size=size, player_id=2)
    turn = 0
    max_moves = size * size

    # cumulative time spent by each player thinking (seconds)
    cum_time_p1 = 0.0
    cum_time_p2 = 0.0

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
        move_time = time.time() - move_t0

        # accumulate thinking time for the player who just moved
        if pid == 1:
            cum_time_p1 += move_time
        else:
            cum_time_p2 += move_time

        if move is None:
            return 0, 0, 0.0

        r, c = move
        if not board.place_piece(r, c, pid):
            winner_pid = 2 if pid == 1 else 1
            winner_moves = g1.move_counter if winner_pid == 1 else g2.move_counter
            if winner_moves > 0:
                winner_avg_time = (cum_time_p1 / winner_moves) if winner_pid == 1 else (cum_time_p2 / winner_moves)
            else:
                winner_avg_time = 0.0
            return winner_pid, winner_moves, winner_avg_time

        if board.check_connection(pid):
            winner_moves = g1.move_counter if pid == 1 else g2.move_counter
            if winner_moves > 0:
                winner_avg_time = (cum_time_p1 / winner_moves) if pid == 1 else (cum_time_p2 / winner_moves)
            else:
                winner_avg_time = 0.0
            return pid, winner_moves, winner_avg_time

        turn = 1 - turn

    # draw
    return 0, 0, 0.0

def fitness(
    individuo: Vector,
    n_games: int = 24,
    move_weight: float = 0.5,
    time_weight: float = 0.5,
) -> float:
    """Return score (%) vs opponents on 4x4 and 5x5.

    Score combines win rate and speed of wins using `move_counter` and real
    elapsed time. `move_weight` and `time_weight` control the contribution of
    move-count speed and wall-clock time respectively (both in [0,1]).
    """
    wins = 0
    move_speed_bonus_sum = 0.0
    time_speed_bonus_sum = 0.0

    for game_idx in range(n_games):
        # opponent sampled in a neighbourhood around OPPONENT_CENTER
        opponent = random_vector()
        size = 3 if game_idx % 2 == 0 else 3
        max_moves = size * size

        # Randomize who starts to reduce first-move bias.
        if random.random() < 0.5:
            winner, winner_moves, winner_time = play_match(size, individuo, opponent)
            if winner == 1:
                wins += 1
                move_speed_bonus_sum += (max_moves - winner_moves) / max_moves
                # normalize time: compare average seconds per move to baseline 0.05s
                max_time = 0.05
                time_speed_bonus_sum += max(0.0, (max_time - winner_time) / max_time)
        else:
            winner, winner_moves, winner_time = play_match(size, opponent, individuo)
            if winner == 2:
                wins += 1
                move_speed_bonus_sum += (max_moves - winner_moves) / max_moves
                max_time = 0.05
                time_speed_bonus_sum += max(0.0, (max_time - winner_time) / max_time)

    win_pct = wins / n_games if n_games > 0 else 0.0
    avg_move_speed_bonus = move_speed_bonus_sum / n_games if n_games > 0 else 0.0
    avg_time_speed_bonus = time_speed_bonus_sum / n_games if n_games > 0 else 0.0

    # Final score: win percentage plus weighted move and time bonuses (scaled 0-100).
    final_score = (
        win_pct * 100.0
        + move_weight * avg_move_speed_bonus * 100.0
        + time_weight * avg_time_speed_bonus * 100.0
    )
    #print(individuo,f"score={final_score:.2f}% (wins={wins}/{n_games}",)
    return final_score

def init_population(pop_size: int) -> List[List[float]]:
    return [random_vector() for _ in range(pop_size)]

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
    generations: int = 25,
    pc: float = 0.75,
    pm: float = 0.18,
    seed: int = None,
    top_frac: float = 0.05,
    n_games: int = 24,
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
            fitnesses = pool.starmap(fitness, [(ind, n_games) for ind in pop])

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
    parser.add_argument("--pop", type=int, default=160)
    parser.add_argument("--gen", type=int, default=25)
    parser.add_argument("--pc", type=float, default=0.75)
    parser.add_argument("--pm", type=float, default=0.18)
    parser.add_argument("--n_games", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    print("Iniciando optimización GA...")
    best_overall, best_fit, last_best, last_fit = ga_optimize(
        pop_size=args.pop,
        generations=args.gen,
        pc=args.pc,
        pm=args.pm,
        seed=args.seed,
        n_games=args.n_games,
        top_frac=0.05,
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
