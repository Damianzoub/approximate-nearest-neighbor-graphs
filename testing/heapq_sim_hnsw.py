import heapq

def main():
    items = [
        (0.8,"A"),
        (0.4,"B"),
        (1.2,"C"),
        (0.1,"D"),
        (0.6,"E")
    ]
    heap =[]

    for dist,node in items:
        heapq.heappush(heap,(dist,node))

    while heap:
        dist,node = heapq.heappop(heap)
        print(f"Closeset candidate: {node} with dist={dist}")


if __name__ =="__main__":
    main()