import random
import argparse
import time
from typing import List, Tuple
from solution import Minimax, HexNodeGraph
from board import HexBoard

Vector = List[float]

def random_vector() -> Vector:
    return [random.uniform(1.0, 100.0) for _ in range(6)]

def set_weights(weights: Vector) -> None:
    Minimax.a, Minimax.b, Minimax.c, Minimax.d, Minimax.e, Minimax.f = weights

def play_match(size: int, w1: Vector, w2: Vector) -> int:
    """Play one game and return winner id: 1, 2, or 0 for draw."""
    board = HexBoard(size)
    g1 = HexNodeGraph(size=size, player_id=1)
    g2 = HexNodeGraph(size=size, player_id=2)
    turn = 0
    max_moves = size * size

    for _ in range(max_moves):
        if turn == 0:
            set_weights(w1)
            move = Minimax.preminimax(g1, board)
            pid = 1
        else:
            set_weights(w2)
            move = Minimax.preminimax(g2, board)
            pid = 2

        if move is None:
            return 0

        r, c = move
        if not board.place_piece(r, c, pid):
            return 2 if pid == 1 else 1

        if board.check_connection(pid):
            return pid

        turn = 1 - turn

    return 0

def fitness(individuo: Vector, n_games: int = 30) -> float:
    """Return win percentage x100 vs random opponents on 4x4 and 5x5."""
    wins = 0
    for game_idx in range(n_games):
        opponent = random_vector()
        size = 4 if game_idx % 2 == 0 else 5

        # Randomize who starts to reduce first-move bias.
        if random.random() < 0.5:
            winner = play_match(size, individuo, opponent)
            if winner == 1:
                wins += 1
        else:
            winner = play_match(size, opponent, individuo)
            if winner == 2:
                wins += 1
    print(individuo, wins/n_games * 100.0)
    return wins / n_games * 100.0

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
            ind[i] = max(1.0, min(100.0, ind[i]))

def ga_optimize(
    pop_size: int = 60,
    generations: int = 7,
    pc: float = 0.8,
    pm: float = 0.12,
    seed: int = None,
):
    if seed is not None:
        random.seed(seed)

    pop = init_population(pop_size)

    best_overall: Vector = []
    best_fit_overall = -1.0

    for gen in range(1, generations + 1):
        fitnesses = [fitness(ind, n_games=30) for ind in pop]

        for ind, fit in zip(pop, fitnesses):
            print(f"{fit:6.2f}% -> {[round(x, 4) for x in ind]}")

        ranked = sorted(range(len(pop)), key=lambda i: fitnesses[i], reverse=True)
        best_idx = ranked[0]
        best_fit = fitnesses[best_idx]
        best_ind = pop[best_idx][:]

        if best_fit > best_fit_overall:
            best_fit_overall = best_fit
            best_overall = best_ind[:]

        print(f"Generacion {gen} terminada")

        top_n = max(1, int(0.10 * pop_size))
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

    return best_overall, best_fit_overall

def main():
    parser = argparse.ArgumentParser(description="GA tuner for Minimax weights (a..f) based on match win rate")
    parser.add_argument("--pop", type=int, default=60)
    parser.add_argument("--gen", type=int, default=7)
    parser.add_argument("--pc", type=float, default=0.8)
    parser.add_argument("--pm", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    print("Iniciando optimización GA...")
    ga_optimize(
        pop_size=args.pop,
        generations=args.gen,
        pc=args.pc,
        pm=args.pm,
        seed=args.seed,
    )
    _ = time.time() - t0

if __name__ == "__main__":
    main()
