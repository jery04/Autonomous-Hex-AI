from solution import HexNodeGraph


def test_create_5x5_graph():
    g = HexNodeGraph()
    matrix = g.create_node_matrix(7, orientation=1)
    
    if g.is_any_adjacent_marked(2, 0):
        g.add_adjacents_to_node(2, 0, g.extreme1)
    g.mark_node_at(2, 0)
    
    if g.is_any_adjacent_marked(2, 1):
        g.add_adjacents_to_node(2, 1, g.extreme1)
    g.mark_node_at(2, 1)
    
    if g.is_any_adjacent_marked(2, 2):
        g.add_adjacents_to_node(2, 2, g.extreme1)
    g.mark_node_at(2, 2)
    
    if g.is_any_adjacent_marked(2, 6):
        g.add_adjacents_to_node(2, 6, g.extreme2)
    g.mark_node_at(2, 6)
    
    if g.is_any_adjacent_marked(2, 5):
        g.add_adjacents_to_node(2, 5, g.extreme2)
    g.mark_node_at(2, 5)
    

    d = g.distance_between_extremes()
    
    print("distance_between_extremes:", d)
    

if __name__ == '__main__':
    test_create_5x5_graph()