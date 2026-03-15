from solution import HexNodeGraph


def demo():
    print("Distances (orientation=1):")
    for size in (1, 2, 3, 7):
        g = HexNodeGraph()
        g.create_node_matrix(size, orientation=1)
        print(f" - size={size}: distance = {g.distance_between_extremes()}")

    print("\nBlocked example (size=1):")
    g = HexNodeGraph()
    g.create_node_matrix(1, orientation=1)
    g.remove_node_at(0, 0, verbose=False)
    print(f" - after removing (0,0): distance = {g.distance_between_extremes()}")

    print("\nOrientation=2 example (size=2):")
    g = HexNodeGraph()
    g.create_node_matrix(2, orientation=2)
    print(f" - size=2 orientation=2: distance = {g.distance_between_extremes()}")


if __name__ == '__main__':
    demo()
