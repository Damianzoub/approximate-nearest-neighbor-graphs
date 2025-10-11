import numpy as np 
import faiss 

dimensions = 128
nb = 1000
nq = 3

data = np.random.random((nb,dimensions)).astype("float32")
queries = np.random.random((nq,dimensions)).astype("float32")
print(f"Data: {data}")
print(f"Queries: {queries}")

#HNSW index 
index = faiss.IndexHNSWFlat(dimensions,32)
print(index)
index.hnsw.efConstruction = 100
index.hnsw.efSearch = 50

index.add(data)

k = 5
distances , indices = index.search(queries,k)
print(f"Nearest Neighbors: {indices}")