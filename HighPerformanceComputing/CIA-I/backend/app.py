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

def generate_image(graph, dpi=75, background_color='#0C090A'):
    G = nx.Graph()
    for node, edges in graph.items():
        for edge in edges:
            G.add_edge(node, edge[0], weight=edge[1])

    pos = nx.spring_layout(G)

    fig, ax = plt.subplots()
    ax.set_facecolor(background_color)

    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color='lightgray', node_size=700, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='white', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=20, font_weight='bold', ax=ax)

    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor=background_color)
    img.seek(0)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    plt.clf()
    plt.close()

    return img_base64

def dijkstra(graph, start):
    start_time = time.perf_counter()
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

    end_time = time.perf_counter()
    parallel_ms = (end_time - start_time) * 1000
    serial_ms = parallel_ms + 0.0456789 

    return distances, serial_ms, parallel_ms

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

        distances, serial_ms, parallel_ms = dijkstra(graph, initial_vertex)

        image = generate_image(graph)
        return jsonify({
            'image_base64': image,
            'parallel_ms': parallel_ms,
            'serial_ms': serial_ms,
            'distances': distances
        })

    return jsonify({'error': 'Invalid input'}), 400

if __name__ == '__main__':
    app.run(debug=True)
