import numpy as np
import random
from copy import deepcopy
import time

# Load and initialize graph
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

def assign_partitions(graph: dict):
    vertices = list(graph.keys())
    np.random.shuffle(vertices)
    half_size = len(vertices) // 2
    for i, vertex in enumerate(vertices):
        graph[vertex]["partition"] = 0 if i < half_size else 1
    return graph

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
        for neighbor in data["neighbors"]:
            neighbor_partition = graph[neighbor]["partition"]
            if vertex_partition != neighbor_partition:
                cut_size += 0.5
    return int(cut_size)

def solution_to_binary_string(graph: dict) -> str:
    max_id = max(graph.keys())
    size = len(graph)
    start_idx = 0 if max_id == size - 1 else 1
    binary = ""
    for i in range(start_idx, start_idx + size):
        binary += str(graph[i]["partition"])
    return binary

def binary_string_to_solution(binary: str, graph_template: dict) -> dict:
    graph = deepcopy(graph_template)
    start_idx = 0 if max(graph.keys()) == len(binary) - 1 else 1
    for i, bit in enumerate(binary):
        graph[start_idx + i]["partition"] = int(bit)
    return graph


# Recombination Function
def recombine_parents(parent1: str, parent2: str) -> str:
    p1 = np.array([int(b) for b in parent1])
    p2 = np.array([int(b) for b in parent2])
    l = len(p1)
    hamming_dist = np.sum(p1 != p2)
    if hamming_dist > l / 2:
        p1 = 1 - p1

    child = np.empty(l, dtype=int)
    agree = (p1 == p2)
    child[agree] = p1[agree]

    diff_indices = np.where(~agree)[0]
    fixed_ones = np.sum(child[agree])
    ones_needed = (l // 2) - fixed_ones
    chosen_for_ones = set(np.random.choice(diff_indices, size=ones_needed, replace=False))

    for i in diff_indices:
        child[i] = 1 if i in chosen_for_ones else 0

    return ''.join(map(str, child))


# FM Local Search
class Vertex:
    def __init__(self, vid, neighbors, partition):
        self.id = vid
        self.neighbors = neighbors
        self.partition = partition
        self.gain = 0
        self.locked = False
        self.bucket_prev = None
        self.bucket_next = None
        self.bucket_index = None

def fm_local_search(graph: dict) -> dict:
    vertices = {}
    for vid, data in graph.items():
        vertices[vid] = Vertex(vid, data["neighbors"], data["partition"])

    max_degree = max(len(v.neighbors) for v in vertices.values())
    num_buckets = 2 * max_degree + 1
    offset = max_degree
    buckets = [None] * num_buckets

    def insert(v, gain_val):
        v.gain = gain_val
        idx = gain_val + offset
        v.bucket_index = idx
        v.bucket_prev = None
        v.bucket_next = buckets[idx]
        if buckets[idx] is not None:
            buckets[idx].bucket_prev = v
        buckets[idx] = v

    def remove(v):
        idx = v.bucket_index
        if v.bucket_prev is not None:
            v.bucket_prev.bucket_next = v.bucket_next
        else:
            buckets[idx] = v.bucket_next
        if v.bucket_next is not None:
            v.bucket_next.bucket_prev = v.bucket_prev
        v.bucket_prev = v.bucket_next = None

    def update_gain(v, delta):
        remove(v)
        insert(v, v.gain + delta)

    for v in vertices.values():
        i = sum(1 for nb in v.neighbors if vertices[nb].partition == v.partition)
        e = len(v.neighbors) - i
        insert(v, e - i)

    moves = []
    cumulative_gain = 0
    locked = set()

    for _ in range(len(vertices)):
        selected = None
        for b in range(num_buckets - 1, -1, -1):
            if buckets[b] is not None:
                selected = buckets[b]
                break
        if selected is None:
            break
        remove(selected)
        selected.locked = True
        locked.add(selected.id)
        cumulative_gain += selected.gain
        moves.append((selected.id, selected.gain, cumulative_gain))

        for nb in selected.neighbors:
            if nb in locked:
                continue
            neighbor = vertices[nb]
            delta = 2 if neighbor.partition == selected.partition else -2
            update_gain(neighbor, delta)

        selected.partition = 1 - selected.partition

    best_gain = -float('inf')
    best_index = -1
    for i, (_, _, gain) in enumerate(moves):
        if gain > best_gain:
            best_gain = gain
            best_index = i

    for i in range(len(moves) - 1, best_index, -1):
        vid, _, _ = moves[i]
        vertices[vid].partition = 1 - vertices[vid].partition

    new_graph = deepcopy(graph)
    for vid, v in vertices.items():
        new_graph[vid]["partition"] = v.partition
    return new_graph

def run_mls(graph_template: dict, num_restarts: int = 25) -> dict:
    best_graph = None
    best_cut = float('inf')
    start_time = time.time()

    for i in range(num_restarts):
        # Deep copy and random partitioning
        graph = deepcopy(graph_template)
        assign_partitions(graph)

        # Local search
        improved_graph = fm_local_search(graph)
        cut = calculate_cut_size(improved_graph)

        if cut < best_cut:
            best_cut = cut
            best_graph = improved_graph

        print(f"[MLS {i+1}/{num_restarts}] Cut size: {cut}")

    total_time = time.time() - start_time
    print(f"\nBest cut found: {best_cut}")
    print(f"Time elapsed: {total_time:.2f} seconds")

    return {
        "best_graph": best_graph,
        "cut_size": best_cut,
        "time": total_time
    }

#here we start the ILS part

def perturb_sol(graph: dict, k: int) -> dict:
    graph = deepcopy(graph)
    vertices = list(graph.keys())
    to_flip = random.sample(vertices, k)
    
    for i in to_flip:
        graph[i]["partition"] = 1 - graph[i]["partition"]
    return graph

def run_ils(graph_template: dict, k: int, num_restarts: int=25) -> dict:
    
    start_time = time.time()
    same_optimum_count = 0
    
    graph = assign_partitions(deepcopy(graph_template))
    current = fm_local_search(graph) #isnt this always the case for ils?
    best = deepcopy(current)
    best_cut = calculate_cut_size(current)
    
    print(f"[ILS Init] Cut size: {best_cut}")
    
    
    for i in range(num_restarts):
        perturbed = perturb_sol(current, k)
        local_optimum = fm_local_search(perturbed)
        cut = calculate_cut_size(local_optimum)
        
        if calculate_cut_size(current) == cut:
            same_optimum_count += 1
            
        if cut < best_cut:
            best = deepcopy(local_optimum)
            best_cut = cut
            current = deepcopy(local_optimum)
        
        print(f"[ILS {i+1}/{num_restarts}] Cut size: {cut}")
    
    total_time = time.time() - start_time
    print(f"\nBest cut found (ILS): {best_cut}")
    print(f"Time elapsed: {total_time:.2f} seconds")
    print(f"Same local opt found {same_optimum_count}/{num_restarts} times")

    return {
        "best_graph": best,
        "cut_size": best_cut,
        "time": total_time,
        "same_optimum_count": same_optimum_count
    }


#Test Run
if __name__ == "__main__":
    parsed_graph = load_graph("Graph500.txt")
    partitioned_graph1 = assign_partitions(deepcopy(parsed_graph))
    partitioned_graph2 = assign_partitions(deepcopy(parsed_graph))

    graph_bin1 = solution_to_binary_string(partitioned_graph1)
    graph_bin2 = solution_to_binary_string(partitioned_graph2)

    print("Initial cut size:", calculate_cut_size(partitioned_graph1))
    print("Partitions (0s and 1s):", count_partitions(partitioned_graph1))

    child_binary = recombine_parents(graph_bin1, graph_bin2)
    print("Child solution:", child_binary)

    child_graph = binary_string_to_solution(child_binary, parsed_graph)
    improved_graph = fm_local_search(child_graph)

    print("Improved cut size:", calculate_cut_size(improved_graph))
    print("Partitions after FM:", count_partitions(improved_graph))
    mls_result = run_mls(parsed_graph, num_restarts=25)

    print("\nFinal partition sizes:", count_partitions(mls_result["best_graph"]))
    print("Final cut size:", mls_result["cut_size"])

    #15 is k here,
    ils_result = run_ils(parsed_graph, 15, num_restarts = 25)
    #we could also do this, I have commented it for now
    #ils_result= {}
    
    #for k in [1, 2, 5, 10, 15, 20]:
    #    print("\nRunning ILS with k = {k}")
    #    current_result = run_ils(parsed_graph, k=k, num_restarts=25)
    #    ils_result[k] = current_result
    
    print("\nFinal partition sizes:", count_partitions(ils_result["best_graph"]))
    print("Final cut size:", ils_result["cut_size"])