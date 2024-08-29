import random

def generate_random_graph(num_vertices, max_edges_per_vertex, max_weight=100):
    graph = {i: [] for i in range(num_vertices)}
    
    for i in range(num_vertices):
        # Ensure each vertex has at least one edge
        if not graph[i]:
            neighbor = random.choice([n for n in range(num_vertices) if n != i])
            weight = random.randint(1, max_weight)
            graph[i].append((neighbor, weight))
            graph[neighbor].append((i, weight))
        
        # Add additional edges up to max_edges_per_vertex
        num_edges = random.randint(1, max_edges_per_vertex)
        neighbors = random.sample([n for n in range(num_vertices) if n != i], num_edges)
        
        for neighbor in neighbors:
            if neighbor != i and neighbor not in [n for n, _ in graph[i]]:
                weight = random.randint(1, max_weight)
                graph[i].append((neighbor, weight))
                graph[neighbor].append((i, weight))  # Assuming an undirected graph
    
    return graph

# Example usage:
num_vertices = 100
max_edges_per_vertex = 4
graph = generate_random_graph(num_vertices, max_edges_per_vertex)
print(graph)
