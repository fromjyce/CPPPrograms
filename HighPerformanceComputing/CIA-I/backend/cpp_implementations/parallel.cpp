#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <map>
#include <string>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <omp.h>

using namespace std;
using namespace std::chrono;

extern "C" {
    // Function to parse the graph string
    void parseGraph(const char* graphStr, map<string, vector<pair<string, int>>>& graph, string& startNode);

    // Dijkstra's Algorithm - Parallel version
    extern void dijkstra(const map<string, vector<pair<string, int>>>& graph, const string& startNode, string& result, long long& timeTaken);
}

// Function to parse the graph string
void parseGraph(const char* graphStr, map<string, vector<pair<string, int>>>& graph, string& startNode) {
    stringstream ss(graphStr);
    string line;
    while (getline(ss, line, '\n')) {
        size_t pos = line.find(':');
        if (pos != string::npos) {
            string node = line.substr(0, pos);
            string edgesStr = line.substr(pos + 1);
            vector<pair<string, int>> edges;
            stringstream edgesStream(edgesStr);
            string edge;
            while (getline(edgesStream, edge, ',')) {
                size_t dashPos = edge.find('-');
                if (dashPos != string::npos) {
                    string targetNode = edge.substr(0, dashPos);
                    int weight = stoi(edge.substr(dashPos + 1));
                    edges.push_back(make_pair(targetNode, weight));
                }
            }
            graph[node] = edges;
        } else if (line.find("start:") == 0) {
            startNode = line.substr(6);
        }
    }
}

// Function to find the node with the minimum distance
string minDistance(const map<string, int>& dist, const vector<string>& nodes) {
    string minNode;
    int minDist = numeric_limits<int>::max();
    
    for (const string& node : nodes) {
        if (dist.at(node) < minDist) {
            minDist = dist.at(node);
            minNode = node;
        }
    }
    return minNode;
}

// Dijkstra's Algorithm - Parallel version
void dijkstra(const map<string, vector<pair<string, int>>>& graph, const string& startNode, string& result, long long& timeTaken) {
    // Set the number of threads (for example, 4)
    int numThreads = 4;
    omp_set_num_threads(numThreads);

    map<string, int> dist;
    map<string, string> prev;
    vector<string> nodes;

    for (const auto& pair : graph) {
        string node = pair.first;
        nodes.push_back(node);
        dist[node] = numeric_limits<int>::max();
        prev[node] = "";
    }

    dist[startNode] = 0;

    auto start = high_resolution_clock::now();

    while (!nodes.empty()) {
        string u = minDistance(dist, nodes);
        nodes.erase(remove(nodes.begin(), nodes.end(), u), nodes.end());

        #pragma omp parallel
        {
            #pragma omp for
            for (size_t i = 0; i < graph.at(u).size(); ++i) {
                string v = graph.at(u)[i].first;
                int weight = graph.at(u)[i].second;
                if (dist[u] != numeric_limits<int>::max() && dist[u] + weight < dist[v]) {
                    #pragma omp critical
                    {
                        if (dist[u] + weight < dist[v]) {
                            dist[v] = dist[u] + weight;
                            prev[v] = u;
                        }
                    }
                }
            }
        }
    }

    auto end = high_resolution_clock::now();
    timeTaken = duration_cast<microseconds>(end - start).count();

    stringstream resultStream;
    for (const auto& pair : dist) {
        resultStream << "Distance from " << startNode << " to " << pair.first << " is " << pair.second << "\n";
    }
    result = resultStream.str();
}