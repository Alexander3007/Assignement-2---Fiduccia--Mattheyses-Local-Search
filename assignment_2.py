import numpy as np


#Loading graoph from text file and creating dict
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

# Random initial partition assignement
def assign_partitions(graph: dict):
    vertices = list(graph.keys())
    np.random.shuffle(vertices)

    # Divide vertices evenly between partitions
    half_size = len(vertices) // 2
    
    for i, vertex in enumerate(vertices):
        # First half gets partition 0, second half gets partition 1
        graph[vertex]["partition"] = 0 if i < half_size else 1
    
    return graph 

#count the number of vertices in each partition(should be equal size)
def count_partitions(graph: dict):
    partition_counts = {0: 0, 1: 0}
    for vertex in graph:
        partition = graph[vertex]["partition"]
        partition_counts[partition] += 1
    return partition_counts

def calculate_cut_size(graph: dict) -> int:
    cut_size = 0
    
    for vertex, data in graph.items():
        vertex_partition = data["partition"]
        
        # Check neighbor
        for neighbor in data["neighbors"]:
            neighbor_partition = graph[neighbor]["partition"]
            
            # If vertex and neighbor are in different partitions edge crosses the cut
            if vertex_partition != neighbor_partition:
                # Increment by 0.5 because each edge will be counted twice otherwise
                cut_size += 0.5
                
    return int(cut_size)



def solution_to_binary_string(graph: dict) -> str:
    """Convert partition solution to a binary string."""
    max_id = max(graph.keys())
    size = len(graph)

    start_idx = 0 if max_id == size-1 else 1
    
    binary = ""
    for i in range(start_idx, start_idx + size):
        binary += str(graph[i]["partition"])
    
    return binary


def recombination(parent1 :str, parent2: str) -> str:

    ham_dist = 0

    for char in parent1:
        for char in parent2:
            if parent1[char] != parent2[char]:
                ham_dist += 1

    print(ham_dist)


parsed_graph = load_graph("Graph500.txt")
partitioned_graph1 = assign_partitions(parsed_graph)
partitioned_graph2 = assign_partitions(parsed_graph)
# Print sample of vertices and their data
for v, data in list(partitioned_graph1.items())[:5]:  
    print(f"Vertex {v}: {data}")

solution_to_binary_string(partitioned_graph1) 
solution_to_binary_string(partitioned_graph2) 
recombination(partitioned_graph1, partitioned_graph2)
# Count and print the number of vertices in each partition
#partition_counts = count_partitions(partitioned_graph)
print("\nPartition counts:")
#print(f"Partition 1: {partition_counts[0]} vertices")
#print(f"Partition 2: {partition_counts[1]} vertices")
cut_size = calculate_cut_size(partitioned_graph1)
print(f"Cut size: {cut_size} edges")




