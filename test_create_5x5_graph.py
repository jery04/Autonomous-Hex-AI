from solution import HexNodeGraph


def test_create_5x5_graph():
    b1 = HexNodeGraph()
    matrix = b1.create_node_matrix(5, orientation=2)
    
    # marcar
    b1.mark_node_at(0, 2)
    b1.mark_node_at(1, 2)
    b1.mark_node_at(2, 1)
    b1.mark_node_at(2, 2)
    b1.mark_node_at(2, 3)
    b1.mark_node_at(3, 0)
  
    print(b1.matrix[2][0].neighbors)
    
    # borrar
    #b1.remove_node_at(1,3)
    #b1.remove_node_at(2,0)
    #b1.remove_node_at(3,1)
    #b1.remove_node_at(3,2)
    #b1.remove_node_at(3,3)
    
    
    
    
    #d1 = b1.distance_between_extremes()
    #print("distance_between_extremes:", d1)
    

if __name__ == '__main__':
    test_create_5x5_graph()