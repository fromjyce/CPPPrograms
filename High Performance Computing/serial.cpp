#include <iostream>
#include <vector>
#include <queue>
#include <climits>

using namespace std;

typedef pair<int, int> pii; // (weight, vertex)

// Function to perform Dijkstra's algorithm
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

    // Run Dijkstra's algorithm
    vector<int> distances = dijkstra(num_vertices, graph, source);

    // Output the distances
    for (int i = 0; i < num_vertices; ++i) {
        cout << "Distance from vertex " << source << " to vertex " << i << " is " << distances[i] << endl;
    }

    return 0;
}
