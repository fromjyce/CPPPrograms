#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <omp.h>

using namespace std;

typedef pair<int, int> pii; // (weight, vertex)


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
    int num_vertices = 100; // Example number of vertices
    int source = 0; // Example source vertex

    // Graph represented as an adjacency list
    vector<vector<pii>> graph(num_vertices);

    // Example graph construction (you can replace this with the random graph generator)
    graph[0].push_back({1, 4});
    graph[0].push_back({2, 1});
    graph[1].push_back({3, 1});
    graph[2].push_back({1, 2});
    graph[2].push_back({3, 5});
    graph[3].push_back({4, 3});

    // Run parallel Dijkstra's algorithm
    vector<int> distances = dijkstra_parallel(num_vertices, graph, source);

    // Output the distances
    for (int i = 0; i < num_vertices; ++i) {
        cout << "Distance from vertex " << source << " to vertex " << i << " is " << distances[i] << endl;
    }

    return 0;
}
