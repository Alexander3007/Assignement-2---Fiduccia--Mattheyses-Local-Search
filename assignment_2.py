import numpy as np

def load_graph(filename: str) -> dict:
    graph = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            vertex_id = int(parts[0])  
            degree = int(parts[2])  
            connected_vertices = list(map(int, parts[3:])) 
            graph[vertex_id] = {
                "degree": degree,
                "neighbors": connected_vertices,
                "partition": 0  
            }
    return graph

# Random initial partition assignement (probably should make sure that they are of equal size)
def assign_random_partitions(graph: dict):
    for vertex in graph:
        graph[vertex]["partition"] = np.random.choice([1, 2])  

    return graph  


parsed_graph = load_graph("Graph500.txt")
partitioned_graph = assign_random_partitions(parsed_graph)

for v, data in list(partitioned_graph.items())[:50]:  
    print(f"Vertex {v}: {data}")
