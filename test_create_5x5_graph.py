from solution import HexNodeGraph
from board import HexBoard

def test_create_5x5_graph():
    b1 = HexNodeGraph(5,1)
    board = HexBoard(5)
    board.place_piece(1, 2, 2)
    print("move:", b1.detect_opponent_move(board))
    
    # marcar
    b1.mark_node_at(1, 4, 1)
    b1.mark_node_at(2, 0, 1)
    b1.mark_node_at(3, 1, 1)
    b1.mark_node_at(3, 2, 1)
    b1.mark_node_at(3, 4, 1)
    
    
    b1.mark_node_at(0, 3, 2)
    b1.mark_node_at(1, 3, 2)
    b1.mark_node_at(2, 3, 2)
    b1.mark_node_at(2, 2, 2)
    b1.mark_node_at(3, 3, 2)
    print(b1.is_cell_available(1, 4))
    d1 = b1.distance_between_extremes(1)
    print("distance_between_extremes:", d1)
    

if __name__ == '__main__':
    test_create_5x5_graph()