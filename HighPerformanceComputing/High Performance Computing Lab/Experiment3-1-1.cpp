#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>

using namespace std;
using namespace std::chrono;


void fillMatrix(int* matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = rand() % 10;
    }
}


void method_one(const int* A, const int* B, int* C, int n) {
    memset(C, 0, sizeof(int) * n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}


void method_two(const int* A, const int* B, int* C, int n) {
    memset(C, 0, sizeof(int) * n * n);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}


void method_three(const int* A, const int* B, int* C, int n) {
    memset(C, 0, sizeof(int) * n * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}


void method_four(const int* A, const int* B, int* C, int n) {
    memset(C, 0, sizeof(int) * n * n);
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < n; ++i) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}


void method_five(const int* A, const int* B, int* C, int n) {
    memset(C, 0, sizeof(int) * n * n);
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}


void method_six(const int* A, const int* B, int* C, int n) {
    memset(C, 0, sizeof(int) * n * n);
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

int main() {
    int n = 3000;

    int* A = (int*)malloc(n * n * sizeof(int));
    int* B = (int*)malloc(n * n * sizeof(int));
    int* C = (int*)malloc(n * n * sizeof(int));
    fillMatrix(A, n);
    fillMatrix(B, n);

    
    auto start_one = high_resolution_clock::now();
    method_one(A, B, C, n);
    auto end_one = high_resolution_clock::now();
    auto duration_one = duration_cast<milliseconds>(end_one - start_one);
    cout << "Time taken for Method One: " << duration_one.count() << " milliseconds" << endl;

    auto start_two = high_resolution_clock::now();
    method_two(A, B, C, n);
    auto end_two = high_resolution_clock::now();
    auto duration_two = duration_cast<milliseconds>(end_two - start_two);
    cout << "Time taken for Method Two: " << duration_two.count() << " milliseconds" << endl;

    auto start_three = high_resolution_clock::now();
    method_three(A, B, C, n);
    auto end_three = high_resolution_clock::now();
    auto duration_three = duration_cast<milliseconds>(end_three - start_three);
    cout << "Time taken for Method Three: " << duration_three.count() << " milliseconds" << endl;

    auto start_four = high_resolution_clock::now();
    method_four(A, B, C, n);
    auto end_four = high_resolution_clock::now();
    auto duration_four = duration_cast<milliseconds>(end_four - start_four);
    cout << "Time taken for Method Four: " << duration_four.count() << " milliseconds" << endl;

    
    auto start_five = high_resolution_clock::now();
    method_five(A, B, C, n);
    auto end_five = high_resolution_clock::now();
    auto duration_five = duration_cast<milliseconds>(end_five - start_five);
    cout << "Time taken for Method Five: " << duration_five.count() << " milliseconds" << endl;

    
    auto start_six = high_resolution_clock::now();
    method_six(A, B, C, n);
    auto end_six = high_resolution_clock::now();
    auto duration_six = duration_cast<milliseconds>(end_six - start_six);
    cout << "Time taken for Method Six: " << duration_six.count() << " milliseconds" << endl;

    free(A);
    free(B);
    free(C);

    return 0;
}
