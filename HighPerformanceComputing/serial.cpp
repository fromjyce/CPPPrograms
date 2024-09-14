#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include <chrono>
#include <map>
#include <string>
#include <sstream>
#include <cstring>

using namespace std;
using namespace std::chrono;

extern "C" {
    void parseGraph(const char* graphStr, map<string, vector<pair<string, int>>>& graph, string& startNode);
    extern void dijkstra(const map<string, vector<pair<string, int>>>& graph, const string& startNode, string& result, long long& timeTaken);
}

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


// Dijkstra's Algorithm - Serial version
void dijkstra(const map<string, vector<pair<string, int>>>& graph, const string& startNode, string& result, long long& timeTaken) {
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

        for (const auto& neighbor : graph.at(u)) {
            string v = neighbor.first;
            int weight = neighbor.second;
            if (dist[u] != numeric_limits<int>::max() && dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                prev[v] = u;
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
