#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <map>
#include <string>
#include <algorithm>
#include <omp.h>

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

        #pragma omp parallel //directive begins a parallel region, allowing multiple threads to execute the loop in parallel
        {
            #pragma omp for schedule(dynamic, 1) //The dynamic, 1 schedule means each thread will take one iteration of the loop at a time, dynamically allocating iterations as threads become available.
            for (size_t i = 0; i < graph.at(u).size(); ++i) {
                T v = graph.at(u)[i].first;
                int weight = graph.at(u)[i].second;
                int newDist;
                if (dist[u] != numeric_limits<int>::max()) {
                    newDist = dist[u] + weight;
                    #pragma omp critical //Ensures only one thread at a time accesses the following code block, avoiding race conditions.
                    {
                        if (newDist < dist[v]) {
                            dist[v] = newDist;
                            prev[v] = u;
                        }
                    }
                }
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
    
    chrono::duration<double, milli> duration = end - start;
    cout << "Parallel version took " << duration.count() << " milliseconds." << endl;

    return 0;
}