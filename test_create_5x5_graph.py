from solution import HexNodeGraph, Minimax
from board import HexBoard
import time

def test_create_5x5_graph():
    b1 = HexNodeGraph(5,2)
    
    # marcar
    b1.mark_node_at(0, 3, 2)
    b1.mark_node_at(2, 3, 1)
    b1.mark_node_at(1, 3, 2)
    b1.mark_node_at(2, 2, 1)
    b1.mark_node_at(1, 2, 2)
    b1.mark_node_at(2, 1, 1)
    b1.mark_node_at(2, 0, 2)
    b1.mark_node_at(1, 1, 1)
    b1.mark_node_at(1, 4, 2)
    

    d1,d2 = b1.distance_between_extremes(1)
    print("distance_between_extremes:", d1, d2)
    #print("position:", d2)

    #print(b1.move_counter)
    #print(b1.territorial_control(1,4, 1, True))
    #start = time.perf_counter()
    #components, max = b1.count_components(1)
    #print("count_components:", components, "max component size:", max)
    #elapsed = time.perf_counter() - start
    #print(f"count_components result: {components}  time: {elapsed:.6f}s")
    
    moves = set()
    for r in range(5):
        for c in range(5):
            if b1.is_cell_available(r,c):
                moves.add((r,c))
    
    print(Minimax.calculate_heuristic(b1, moves))
    #runs = 200
    #total_time = 0.0
    #result = None
    #for _ in range(runs):
    #    start = time.perf_counter()
    #    result = b1.count_threatened_free_nodes(2, moves, True)
    #    total_time += time.perf_counter() - start

    #avg = total_time / runs
    #print(f"count_threatened_free_nodes result: {result}  avg time over {runs} runs: {avg:.6f}s")

if __name__ == '__main__':
    test_create_5x5_graph()