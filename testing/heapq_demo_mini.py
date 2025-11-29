import heapq 
#we can ee that the last node that enters it is leaving also first
def main():
    heap =[]

    heapq.heappush(heap,(5,"node_A"))
    heapq.heappush(heap,(2,"node_B"))
    heapq.heappush(heap,(7,"node_C"))
    heapq.heappush(heap,(1,"node_D"))

    print("Heap structure: ",heap )

    dist,name=  heapq.heappop(heap)
    print("Popped: ",dist, name)
    dist,name=  heapq.heappop(heap)
    print("Popped: ",dist, name)
    dist,name=  heapq.heappop(heap)
    print("Popped: ",dist, name)
    dist,name=  heapq.heappop(heap)
    print("Popped: ",dist, name)
if __name__ == "__main__":
    main()