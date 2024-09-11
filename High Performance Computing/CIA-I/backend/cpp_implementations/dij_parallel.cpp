#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <omp.h>
#include <chrono>

using namespace std;

typedef pair<int, int> pii;
vector<int> dijkstra_parallel(int num_vertices, const vector<vector<pii>>& graph, int source) {
    vector<int> dist(num_vertices, INT_MAX);
    vector<bool> visited(num_vertices, false);
    priority_queue<pii, vector<pii>, greater<pii>> pq;

    dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        int current_vertex = -1;

        #pragma omp parallel
        {
            int local_vertex = -1;

            #pragma omp for nowait
            for (int i = 0; i < num_vertices; ++i) {
                if (!visited[i] && (local_vertex == -1 || dist[i] < dist[local_vertex])) {
                    local_vertex = i;
                }
            }

            #pragma omp critical
            {
                if (current_vertex == -1 || (local_vertex != -1 && dist[local_vertex] < dist[current_vertex])) {
                    current_vertex = local_vertex;
                }
            }
        }

        if (current_vertex == -1) {
            break;
        }

        visited[current_vertex] = true;

        #pragma omp parallel for
        for (int i = 0; i < graph[current_vertex].size(); ++i) {
            int neighbor = graph[current_vertex][i].first;
            int weight = graph[current_vertex][i].second;

            #pragma omp critical
            {
                if (dist[current_vertex] + weight < dist[neighbor]) {
                    dist[neighbor] = dist[current_vertex] + weight;
                    pq.push({dist[neighbor], neighbor});
                }
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
    vector<int> distances = dijkstra_parallel(num_vertices, graph, source);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Time taken by parallel Dijkstra's algorithm: " << duration.count() << " seconds" << endl;
    for (int i = 0; i < num_vertices; ++i) {
        cout << "Distance from vertex " << source << " to vertex " << i << " is " << distances[i] << endl;
    }

    return 0;
}
