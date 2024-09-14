from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

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

@app.route('/api/graph', methods=['POST'])
def process_graph():
    data = request.get_json()

    print(data)

    num_vertices = data.get('vertices', '')
    num_edges = data.get('edges', '')
    initial_vertex = convert_vertex(data.get('initial_vertex', ''))

    print(num_vertices, num_edges, initial_vertex)
    graph = {}

    vertices_are_chars = isinstance(initial_vertex, str)
    initial_vertex = convert_instance(initial_vertex)

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

        print(graph)

        graph_str = graph_to_string(graph)
        print(graph_str)
    return jsonify({'error': 'Invalid input'}), 400

if __name__ == '__main__':
    app.run(debug=True)