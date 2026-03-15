from solution import HexNodeGraph
import time


def main():
    g = HexNodeGraph()
    g.create_node_matrix(5, orientation=1)

    t0 = time.perf_counter()
    d1 = g.distance_between_extremes()
    t1 = time.perf_counter()
    print("distance (stop_on_first=True):", d1)
    print(f"time (stop_on_first=True): {t1 - t0:.9f} seconds")
    
    t2 = time.perf_counter()
    d2 = g.distance_between_extremes(stop_on_first=False)
    t3 = time.perf_counter()

    print("distance (stop_on_first=False):", d2)
    print(f"time (stop_on_first=False): {t3 - t2:.9f} seconds")


if __name__ == '__main__':
    main()
