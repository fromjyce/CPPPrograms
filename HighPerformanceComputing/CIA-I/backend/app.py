from flask import Flask, request, jsonify
import time
import ctypes
from ctypes import c_int, POINTER, Structure
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import io
import heapq
import base64
matplotlib.use("Agg")
from flask_cors import CORS

class CPath(Structure):
    _fields_ = [("nodes", c_int * 100),
                ("length", c_int)]

app = Flask(__name__)
CORS(app)

def graph_to_string(graph_dict):
    result = []
    for node, edges in graph_dict.items():
        edges_str = ', '.join(f"('{neighbor}', {weight})" for neighbor, weight in edges)
        result.append(f"'{node}': [{edges_str}]")
    return '{' + ', '.join(result) + '}'

def convert_vertex(vertex):
    try:
        return int(vertex)
    except ValueError:
        return vertex.strip()

def convert_instance(vertex):
    return ord(vertex) if isinstance(vertex, str) else vertex

def generate_image(graph):
    G = nx.Graph()
    for node, edges in graph.items():
        for edge in edges:
            G.add_edge(node, edge[0], weight=edge[1])

    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color='lightgray', node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')

    nx.draw_networkx_labels(G, pos, font_size=15, font_weight='bold')

    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    plt.clf()
    plt.close()

    return img_base64

def dijkstra(graph, start):
    start_time = time.time()
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance > distances[current_node]:
            continue
        if current_node not in graph:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000

    return distances, elapsed_time_ms

@app.route('/api/graph', methods=['POST'])
def process_graph():
    path_nodes = (c_int * 100)()
    path_length = c_int()
    data = request.get_json()
    num_vertices = data.get('vertices', '')
    num_edges = data.get('edges', '')
    initial_vertex = data.get('initial_vertex', '').strip()
    graph = {}

    if num_vertices and num_edges:
        num_vertices = int(num_vertices)
        num_edges = int(num_edges)

        for i in range(num_edges):
            start_vertex = data.get(f'start_vertex_{i}', '')
            end_vertex = data.get(f'end_vertex_{i}', '')
            edge_weight = int(data.get(f'edge_weight_{i}', 0))

            if start_vertex not in graph:
                graph[start_vertex] = []
            if end_vertex not in graph:
                graph[end_vertex] = []

            graph[start_vertex].append((end_vertex, edge_weight))
            graph[end_vertex].append((start_vertex, edge_weight))
        
        print("Graph structure:", graph)
        print("Initial vertex:", initial_vertex)

        distances, elapsed_time_ms = dijkstra(graph, initial_vertex)

        print(distances, elapsed_time_ms)

        image = generate_image(graph)
        return jsonify({
            'image_base64': image,
            'dijkstra_time_ms': elapsed_time_ms,
            'distances': distances
        })

    return jsonify({'error': 'Invalid input'}), 400

if __name__ == '__main__':
    app.run(debug=True)
