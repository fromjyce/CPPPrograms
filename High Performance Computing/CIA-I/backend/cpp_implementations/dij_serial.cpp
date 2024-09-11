#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <chrono>

using namespace std;

typedef pair<int, int> pii;
vector<int> dijkstra(int num_vertices, const vector<vector<pii>>& graph, int source) {
    vector<int> dist(num_vertices, INT_MAX);
    priority_queue<pii, vector<pii>, greater<pii>> pq;

    dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        int current_dist = pq.top().first;
        int current_vertex = pq.top().second;
        pq.pop();

        if (current_dist > dist[current_vertex]) {
            continue;
        }

        for (const auto& edge : graph[current_vertex]) {
            int neighbor = edge.first;
            int weight = edge.second;

            if (dist[current_vertex] + weight < dist[neighbor]) {
                dist[neighbor] = dist[current_vertex] + weight;
                pq.push({dist[neighbor], neighbor});
            }
        }
    }

    return dist;
}

int main() {
    int num_vertices, num_edges;
    cout << "Enter the number of vertices: ";
    cin >> num_vertices;
    cout << "Enter the number of edges: ";
    cin >> num_edges;
    vector<vector<pii>> graph(num_vertices);

    cout << "Enter the edges in the format (u v w) where u and v are vertices (0-indexed) and w is the weight:\n";
    for (int i = 0; i < num_edges; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        graph[u].push_back({v, w});
        graph[v].push_back({u, w});
    }

    int source;
    cout << "Enter the source vertex: ";
    cin >> source;
    auto start = chrono::high_resolution_clock::now();
    vector<int> distances = dijkstra(num_vertices, graph, source);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Time taken by serial Dijkstra's algorithm: " << duration.count() << " seconds" << endl;
    for (int i = 0; i < num_vertices; ++i) {
        cout << "Distance from vertex " << source << " to vertex " << i << " is " << distances[i] << endl;
    }

    return 0;
}
