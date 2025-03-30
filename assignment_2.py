import numpy as np
import random
from copy import deepcopy
import time
import pandas as pd
import matplotlib.pyplot as plt

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

    partition_count = {0: 0, 1: 0}
    for v in vertices.values():
        partition_count[v.partition] += 1

    moves = []
    cumulative_gain = 0
    locked = set()
    total_vertices = len(vertices)
    target_partition_size = total_vertices // 2

    for _ in range(len(vertices)):
        selected = None

        for b in range(num_buckets - 1, -1, -1):
            candidate = buckets[b]
            while candidate:
                pid = candidate.partition
                if partition_count[pid] > target_partition_size:
                    selected = candidate
                    break
                candidate = candidate.bucket_next
            if selected:
                break

        if selected is None:
            break

        remove(selected)
        selected.locked = True
        locked.add(selected.id)
        cumulative_gain += selected.gain
        moves.append((selected.id, selected.gain, cumulative_gain))

        # Update partition count before flipping
        old_partition = selected.partition
        new_partition = 1 - old_partition
        partition_count[old_partition] -= 1
        partition_count[new_partition] += 1

        for nb in selected.neighbors:
            if nb in locked:
                continue
            neighbor = vertices[nb]
            delta = 2 if neighbor.partition == selected.partition else -2
            update_gain(neighbor, delta)

        selected.partition = new_partition

    best_gain = -float('inf')
    best_index = -1
    for i, (_, _, gain) in enumerate(moves):
        if gain > best_gain:
            best_gain = gain
            best_index = i

    for i in range(len(moves) - 1, best_index, -1):
        vid, _, _ = moves[i]
        v = vertices[vid]
        old_partition = v.partition
        new_partition = 1 - old_partition
        v.partition = new_partition
        partition_count[old_partition] -= 1
        partition_count[new_partition] += 1

    new_graph = deepcopy(graph)
    for vid, v in vertices.items():
        new_graph[vid]["partition"] = v.partition
    return new_graph


def run_mls(graph_template: dict, num_restarts: int = 10000, time_limit: float = None) -> dict:
    best_graph = None
    best_cut = float('inf')
    start_time = time.time()
    fm_calls = 0

    for i in range(num_restarts):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break
        graph = deepcopy(graph_template)
        assign_partitions(graph)
        improved_graph = fm_local_search(graph)
        fm_calls += 1
        cut = calculate_cut_size(improved_graph)
        if cut < best_cut:
            best_cut = cut
            best_graph = improved_graph
        #print(f"[MLS {i+1}/{num_restarts}] Cut size: {cut}")
        
    total_time = time.time() - start_time
    print(f"\nBest cut found: {best_cut}")
    print(f"Time elapsed: {total_time:.2f} seconds")

    return {
        "best_graph": best_graph,
        "cut_size": best_cut,
        "time": total_time,
        "fm_calls": fm_calls
    }


#here we start the ILS part

def perturb_sol(graph: dict, k: int) -> dict:
    graph = deepcopy(graph)
    partition_0 = [v for v in graph if graph[v]["partition"] == 0]
    partition_1 = [v for v in graph if graph[v]["partition"] == 1]

    k_half = k // 2
    # Ensure there are enough nodes to flip
    k0 = min(k_half, len(partition_0))
    k1 = min(k - k0, len(partition_1))

    to_flip_0 = random.sample(partition_0, k0)
    to_flip_1 = random.sample(partition_1, k1)

    for v in to_flip_0 + to_flip_1:
        graph[v]["partition"] = 1 - graph[v]["partition"]

    return graph

def run_ils(graph_template: dict, k: int = 15, num_restarts: int = 25, time_limit: float = None) -> dict:
    start_time = time.time()
    same_optimum_count = 0
    
    graph = assign_partitions(deepcopy(graph_template))
    current = fm_local_search(graph)
    best = deepcopy(current)
    best_cut = calculate_cut_size(current)
    fm_calls = 1
    #print(f"[ILS Init] Cut size: {best_cut}")
    
    for i in range(num_restarts):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break
        perturbed = perturb_sol(current, k)
        local_optimum = fm_local_search(perturbed)
        fm_calls += 1
        cut = calculate_cut_size(local_optimum)
        if calculate_cut_size(current) == cut:
            same_optimum_count += 1
        if cut < best_cut:
            best = deepcopy(local_optimum)
            best_cut = cut
            current = deepcopy(local_optimum)
        #print(f"[ILS {i+1}/{num_restarts}] Cut size: {cut}")
        
    total_time = time.time() - start_time
    print(f"\nBest cut found (ILS): {best_cut}")
    print(f"Time elapsed: {total_time:.2f} seconds")
    print(f"Same local opt found {same_optimum_count}/{num_restarts} times")
    
    return {
        "best_graph": best,
        "cut_size": best_cut,
        "time": total_time,
        "same_optimum_count": same_optimum_count,
        "fm_calls": fm_calls
        }

# GLS
def run_gls(graph_template: dict, population_size: int = 50, max_iterations: int = 10000, time_limit: float = None) -> dict:
    start_time = time.time()
    population = []
    fm_calls = 0
    for i in range(population_size):
        graph = deepcopy(graph_template)
        assign_partitions(graph)
        local_optimum = fm_local_search(graph)
        fm_calls += 1
        cut = calculate_cut_size(local_optimum)
        population.append({"graph": local_optimum, "cut": cut})
    
    def select_parents(pop):
        idxs = np.random.choice(len(pop), size=2, replace=False)
        return pop[idxs[0]], pop[idxs[1]]
    
    def get_worst(pop):
        return max(pop, key=lambda ind: ind["cut"])
    
    iterations = 0
    while iterations < max_iterations:
        if time_limit is not None and time.time() - start_time >= time_limit:
            break
        
        parent1_ind, parent2_ind = select_parents(population)
        parent1_str = solution_to_binary_string(parent1_ind["graph"])
        parent2_str = solution_to_binary_string(parent2_ind["graph"])
        child_str = recombine_parents(parent1_str, parent2_str)
        child_graph = binary_string_to_solution(child_str, graph_template)
        child_local = fm_local_search(child_graph)
        fm_calls += 1
        child_cut = calculate_cut_size(child_local)
        worst = get_worst(population)
        if child_cut <= worst["cut"]:
            worst_index = population.index(worst)
            population[worst_index] = {"graph": child_local, "cut": child_cut}
        iterations += 1

    total_time = time.time() - start_time
    best_individual = min(population, key=lambda ind: ind["cut"])
    return {"best_graph": best_individual["graph"], "cut_size": best_individual["cut"], "time": total_time, "fm_calls": fm_calls}

# GLS with ILS combined
def run_gls_ils(graph_template: dict, population_size: int = 50, max_iterations: int = 10000,
                ils_frequency: int = 50, perturbation_size: int = 15, time_limit: float = None) -> dict:
    start_time = time.time()
    population = []
    fm_calls = 0
    for i in range(population_size):
        graph = deepcopy(graph_template)
        assign_partitions(graph)
        local_optimum = fm_local_search(graph)
        fm_calls += 1
        cut = calculate_cut_size(local_optimum)
        population.append({"graph": local_optimum, "cut": cut})
    
    def select_parents(pop):
        idxs = np.random.choice(len(pop), size=2, replace=False)
        return pop[idxs[0]], pop[idxs[1]]
    
    def get_worst(pop):
        return max(pop, key=lambda ind: ind["cut"])
    
    iterations = 0
    while iterations < max_iterations:
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        if iterations % ils_frequency == 0:
            idx = np.random.randint(0, population_size)
            original = population[idx]
            perturbed_graph = perturb_sol(original["graph"], perturbation_size)
            ils_solution = fm_local_search(perturbed_graph)
            fm_calls += 1
            ils_cut = calculate_cut_size(ils_solution)
            if ils_cut <= original["cut"]:
                population[idx] = {"graph": ils_solution, "cut": ils_cut}
        else:
            parent1_ind, parent2_ind = select_parents(population)
            parent1_str = solution_to_binary_string(parent1_ind["graph"])
            parent2_str = solution_to_binary_string(parent2_ind["graph"])
            child_str = recombine_parents(parent1_str, parent2_str)
            child_graph = binary_string_to_solution(child_str, graph_template)
            child_local = fm_local_search(child_graph)
            fm_calls += 1
            child_cut = calculate_cut_size(child_local)
            worst = get_worst(population)
            if child_cut <= worst["cut"]:
                worst_index = population.index(worst)
                population[worst_index] = {"graph": child_local, "cut": child_cut}
        iterations += 1

    total_time = time.time() - start_time
    best_individual = min(population, key=lambda ind: ind["cut"])
    return {"best_graph": best_individual["graph"], "cut_size": best_individual["cut"], "time": total_time, "fm_calls": fm_calls}



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
    
    
    #for k in [1, 2, 5, 10, 15, 20]:
    #    print("\nRunning ILS with k = {k}")
    #    current_result = run_ils(parsed_graph, k=k, num_restarts=25)
    #    ils_result[k] = current_result
    
    print("\nFinal partition sizes:", count_partitions(ils_result["best_graph"]))
    print("Final cut size:", ils_result["cut_size"])

    gls_result = run_gls(parsed_graph, 50)

    print("\nFinal partition sizes:", count_partitions(gls_result["best_graph"]))
    print("Final cut size:", gls_result["cut_size"])

    # Time a single FM pass
    graph_for_fm = assign_partitions(deepcopy(parsed_graph))
    start_fm_time = time.time()
    _ = fm_local_search(graph_for_fm)
    end_fm_time = time.time()
    print(f"\nOne FM pass took {end_fm_time - start_fm_time:.5f} seconds")

# ========= Experiment Wrappers =========

def experiment_fm_passes(graph_template, algorithm_func, fm_limit=10000, runs=25, **kwargs):
    """
    Run a single run of the given algorithm that executes approximately fm_limit FM passes.
    """
    results = []
    for i in range(runs):
        start_time = time.time()
        result = algorithm_func(graph_template, time_limit=None, **kwargs)  # run until num_restarts or iterations complete
        elapsed = time.time() - start_time
        results.append({
            "run": i+1,
            "cut_size": result['cut_size'],
            "time": elapsed,
            "fm_calls": result.get('fm_calls', fm_limit)
        })
        print(f"[FM Run {i+1}] Cut: {result['cut_size']}, FM passes: {result.get('fm_calls', fm_limit)}, Time: {elapsed:.2f}s")
    return pd.DataFrame(results)

def experiment_time_limit(graph_template, algorithm_func, time_limit, runs=25, **kwargs):
    """
    Run a single run of the given algorithm with a specified time limit.
    The algorithm will finish its current FM pass before stopping.
    """
    results = []
    for i in range(runs):
        start_time = time.time()
        result = algorithm_func(graph_template, time_limit=time_limit, **kwargs)
        elapsed = time.time() - start_time
        results.append({
            "run": i+1,
            "cut_size": result['cut_size'],
            "time": elapsed,
            "fm_calls": result.get('fm_calls', None)
        })
        print(f"[Time Run {i+1}] Cut: {result['cut_size']}, FM passes: {result.get('fm_calls', 'N/A')}, Time: {elapsed:.2f}s")
    return pd.DataFrame(results)

# ========= Running the Experiments =========

# First, measure the time for a single MLS run that performs ~10,000 FM passes.
# (We assume run_mls is configured with num_restarts = 10000.)
num_restarts_mls = 10000  # adjust as needed to yield ~10,000 FM passes
print("Measuring MLS run time for 10,000 FM passes...")
mls_single = run_mls(parsed_graph, num_restarts=num_restarts_mls)
time_budget = mls_single["time"]
print(f"MLS run time: {time_budget:.2f} seconds\n")

# Experiment (a): Fixed 10,000 FM passes (each algorithm run is configured to perform ~10,000 FM passes)
mls_fm_df     = experiment_fm_passes(parsed_graph, run_mls, fm_limit=10000, runs=25, num_restarts=num_restarts_mls)
ils_fm_df     = experiment_fm_passes(parsed_graph, run_ils, fm_limit=10000, runs=25, k=15, num_restarts=num_restarts_mls)
gls_fm_df     = experiment_fm_passes(parsed_graph, run_gls, fm_limit=10000, runs=25, population_size=50, max_iterations=10000)
glsils_fm_df  = experiment_fm_passes(parsed_graph, run_gls_ils, fm_limit=10000, runs=25, population_size=50, max_iterations=10000, ils_frequency=50, perturbation_size=15)

# Plot boxplot for FM passes experiment
plt.figure(figsize=(10, 5))
data_fm = [mls_fm_df["cut_size"], ils_fm_df["cut_size"], gls_fm_df["cut_size"], glsils_fm_df["cut_size"]]
plt.boxplot(data_fm, labels=["MLS", "ILS", "GLS", "Hybrid GLS/ILS"])
plt.ylabel("Cut Size")
plt.title("Cut Size Distribution (Fixed 10,000 FM Passes)")
plt.show()

# Experiment (b): Fixed run time (each algorithm gets the same time budget as measured from MLS)
print(f"\nUsing a time budget of {time_budget:.2f} seconds (MLS run time) for the time experiments.\n")
mls_time_df    = experiment_time_limit(parsed_graph, run_mls, time_limit=time_budget, runs=25, num_restarts=num_restarts_mls)
ils_time_df    = experiment_time_limit(parsed_graph, run_ils, time_limit=time_budget, runs=25, k=15, num_restarts=num_restarts_mls)
gls_time_df    = experiment_time_limit(parsed_graph, run_gls, time_limit=time_budget, runs=25, population_size=50, max_iterations=10000)
glsils_time_df = experiment_time_limit(parsed_graph, run_gls_ils, time_limit=time_budget, runs=25, population_size=50, max_iterations=10000, ils_frequency=50, perturbation_size=15)

# Plot boxplot for time experiments
plt.figure(figsize=(10, 5))
data_time = [mls_time_df["cut_size"], ils_time_df["cut_size"], gls_time_df["cut_size"], glsils_time_df["cut_size"]]
plt.boxplot(data_time, labels=["MLS", "ILS", "GLS", "Hybrid GLS/ILS"])
plt.ylabel("Cut Size")
plt.title("Cut Size Distribution (Fixed Run Time from MLS)")
plt.show()

# Optionally, tabulate the mean cut sizes:
print("FM Passes Experiment Means:")
print("MLS:", mls_fm_df["cut_size"].mean())
print("ILS:", ils_fm_df["cut_size"].mean())
print("GLS:", gls_fm_df["cut_size"].mean())
print("Hybrid GLS/ILS:", glsils_fm_df["cut_size"].mean())

print("\nTime Experiment Means:")
print("MLS:", mls_time_df["cut_size"].mean())
print("ILS:", ils_time_df["cut_size"].mean())
print("GLS:", gls_time_df["cut_size"].mean())
print("Hybrid GLS/ILS:", glsils_time_df["cut_size"].mean())

    