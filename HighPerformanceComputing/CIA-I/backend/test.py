from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import ctypes
from ctypes import c_int, POINTER, Structure
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import base64
import os

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

def generate_image(graph, highlight_nodes):
    G = nx.Graph()
    for node, edges in graph.items():
        for edge in edges:
            G.add_edge(node, edge[0])

    pos = nx.spring_layout(G)

    highlight_edges = []
    gray_edges = []

    for edge in G.edges():
        if edge[0] in highlight_nodes and edge[1] in highlight_nodes:
            highlight_edges.append(edge)
        else:
            gray_edges.append(edge)

    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color='lightgray', node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=gray_edges, edge_color='lightgray')

    nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color='lightblue', node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, edge_color='blue')

    nx.draw_networkx_labels(G, pos, font_size=15, font_weight='bold')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    plt.clf()
    plt.close()

    return img_base64

@app.route('/api/graph', methods=['POST'])
def process_graph():
    path_nodes = (c_int * 100)()
    path_length = c_int()
    data = request.get_json()

    num_vertices = data.get('vertices', '')
    num_edges = data.get('edges', '')
    initial_vertex = convert_vertex(data.get('initial_vertex', ''))
    terminal_vertex = convert_vertex(data.get('terminal_vertex', ''))
    algorithm_name = data.get('algorithm_name', 'Unknown Algorithm')
    graph = {}

    vertices_are_chars = isinstance(initial_vertex, str)
    initial_vertex = convert_instance(initial_vertex)
    terminal_vertex = convert_instance(terminal_vertex)

    if num_vertices and num_edges:
        num_vertices = int(num_vertices)
        num_edges = int(num_edges)

        for i in range(num_edges):
            start_vertex = convert_vertex(data.get(f'start_vertex_{i}', ''))
            end_vertex = convert_vertex(data.get(f'end_vertex_{i}', ''))
            edge_weight = int(data.get(f'edge_weight_{i}', 0))

            if start_vertex not in graph:
                graph[start_vertex] = []
            if end_vertex not in graph:
                graph[end_vertex] = []
            graph[start_vertex].append((end_vertex, edge_weight))
            graph[end_vertex].append((start_vertex, edge_weight))

        graph_str = graph_to_string(graph)
        lib = ctypes.CDLL(f'./shared_object_files/{algorithm_name}.so')

        getattr(lib, f'{algorithm_name}_ctypes').argtypes = [ctypes.c_char_p, c_int, c_int, POINTER(c_int), POINTER(c_int)]
        getattr(lib, f'{algorithm_name}_ctypes').restype = c_int
        found = getattr(lib, f'{algorithm_name}_ctypes')(graph_str.encode('utf-8'), initial_vertex, terminal_vertex, path_nodes, ctypes.byref(path_length))

        if found:
            highlight_nodes = [path_nodes[i] for i in range(path_length.value)]
            if vertices_are_chars:
                highlight_nodes = [chr(node) for node in highlight_nodes]
            image = generate_image(graph, highlight_nodes)

            return jsonify({'image_base64': image})
        else:
            return jsonify({'error': 'Target node not found.'}), 404
    return jsonify({'error': 'Invalid input'}), 400

if __name__ == '__main__':
    app.run(debug=True)