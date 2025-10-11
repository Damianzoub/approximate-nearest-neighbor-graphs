import numpy as np 
import faiss 

dimensions = 128
nb = 100000
nq = 1000

data = np.random.random((nb,dimensions)).astype("float32")
queries = np.random.random((nq,dimensions)).astype("float32")


#HNSW index 
index = faiss.IndexHNSWFlat(dimensions,32)
index.hnsw.efConstruction = 100
index.hnsw.efSearch = 50

index.add(data)
k = 5
distances , indices = index.search(queries,k) 
for q in range(5):
    print(f"Query: {q+1}")
    print(f"Indices: {indices[q]}")
    print(f"Distances: {distances[q]}")