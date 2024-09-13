#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <map>
#include <string>
#include <algorithm>

using namespace std;
using namespace std::chrono;

template <typename T>
T minDistance(const map<T, int>& dist, const vector<T>& nodes) {
    T minNode;
    int minDist = numeric_limits<int>::max();
    
    for (const T& node : nodes) {
        if (dist.at(node) < minDist) {
            minDist = dist.at(node);
            minNode = node;
        }
    }
    return minNode;
}

template <typename T>
void dijkstra(const map<T, vector<pair<T, int>>>& graph, const T& startNode) {
    map<T, int> dist;
    map<T, T> prev;
    vector<T> nodes;
    
    for (const auto& pair : graph) {
        T node = pair.first;
        nodes.push_back(node);
        dist[node] = numeric_limits<int>::max();
        prev[node] = T();
    }
    
    dist[startNode] = 0;
    while (!nodes.empty()) {
        T u = minDistance<T>(dist, nodes);
        nodes.erase(remove(nodes.begin(), nodes.end(), u), nodes.end());

        for (const auto& neighbor : graph.at(u)) {
            T v = neighbor.first;
            int weight = neighbor.second;
            if (dist[u] != numeric_limits<int>::max() && dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                prev[v] = u;
            }
        }
    }

    for (const auto& pair : dist) {
        cout << "Distance from " << startNode << " to " << pair.first << " is " << pair.second << endl;
    }
}

int main() {
    map<string, vector<pair<string, int>>> graph = {
        {"A", {{"B", 4}, {"C", 5}}},
        {"B", {{"A", 4}, {"C", 11}, {"D", 7}}},
        {"C", {{"A", 5}, {"B", 11}, {"D", 3}}},
        {"D", {{"B", 7}, {"C", 3}}}
    };
    
    string startNode = "A";
    
    auto start = high_resolution_clock::now();
    dijkstra(graph, startNode);
    auto end = high_resolution_clock::now();
    
    auto duration = duration_cast<microseconds>(end - start).count();
    cout << "Serial version took " << duration << " microseconds." << endl;

    return 0;
}