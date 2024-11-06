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

serial_dijkstra = ctypes.CDLL('./shared_object_files/serial.so')
parallel_dijkstra = ctypes.CDLL('./shared_object_files/parallel.so')

class PathResult(Structure):
    _fields_ = [("distances", c_int * 100), ("num_vertices", c_int)]

serial_dijkstra.dijkstra.restype = PathResult
parallel_dijkstra.dijkstra.restype = PathResult

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

def dijkstra_serial(graph, num_vertices, start):
    graph_c = (c_int * 100 * 100)()
    for i, row in enumerate(graph):
        for j, weight in enumerate(row):
            graph_c[i][j] = weight

    result = serial_dijkstra.dijkstra(graph_c, c_int(num_vertices), c_int(start))
    distances = list(result.distances[:result.num_vertices])
    return distances

def dijkstra_parallel(graph, num_vertices, start):
    graph_c = (c_int * 100 * 100)()
    for i, row in enumerate(graph):
        for j, weight in enumerate(row):
            graph_c[i][j] = weight

    result = parallel_dijkstra.dijkstra(graph_c, c_int(num_vertices), c_int(start))
    distances = list(result.distances[:result.num_vertices])
    return distances

@app.route('/api/graph', methods=['POST'])
def process_graph():
    data = request.get_json()
    num_vertices = data.get('vertices', '')
    num_edges = data.get('edges', '')
    initial_vertex = data.get('initial_vertex', '').strip()
    graph = [[float('inf')] * 100 for _ in range(100)]

    if num_vertices and num_edges:
        num_vertices = int(num_vertices)
        num_edges = int(num_edges)

        for i in range(num_edges):
            start_vertex = int(data.get(f'start_vertex_{i}', ''))
            end_vertex = int(data.get(f'end_vertex_{i}', ''))
            edge_weight = int(data.get(f'edge_weight_{i}', 0))
            graph[start_vertex][end_vertex] = edge_weight
            graph[end_vertex][start_vertex] = edge_weight

        serial_start_time = time.perf_counter()
        distances_serial = dijkstra_serial(graph, num_vertices, int(initial_vertex))
        serial_ms = (time.perf_counter() - serial_start_time) * 1000

        parallel_start_time = time.perf_counter()
        distances_parallel = dijkstra_parallel(graph, num_vertices, int(initial_vertex))
        parallel_ms = (time.perf_counter() - parallel_start_time) * 1000

        image = generate_image(graph)
        return jsonify({
            'image_base64': image,
            'serial_ms': serial_ms,
            'parallel_ms': parallel_ms,
            'distances_serial': distances_serial,
            'distances_parallel': distances_parallel
        })

    return jsonify({'error': 'Invalid input'}), 400

if __name__ == '__main__':
    app.run(debug=True)