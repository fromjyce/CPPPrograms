#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <omp.h>

using namespace std;
using namespace std::chrono;


void matrixVectorProduct(const vector<vector<int>>& A, const vector<int>& B, vector<int>& C) {
    int m = A.size(); 
    int n = A[0].size(); 

    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        C[i] = 0;
        for (int j = 0; j < n; ++j) {
            C[i] += A[i][j] * B[j];
        }
    }
}

void fillMatrix(vector<vector<int>>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = rand() % 10; 
        }
    }
}

void fillVector(vector<int>& vec, int size) {
    for (int i = 0; i < size; ++i) {
        vec[i] = rand() % 10;
    }
}

int main() {
    int n = 100;
    vector<vector<int>> A(n, vector<int>(n));
    vector<int> B(n);
    vector<int> C(n);

    fillMatrix(A, n, n);
    fillVector(B, n);

    auto start = high_resolution_clock::now();
    matrixVectorProduct(A, B, C);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Time taken: " << duration.count() << " milliseconds" << endl;

    return 0;
}

