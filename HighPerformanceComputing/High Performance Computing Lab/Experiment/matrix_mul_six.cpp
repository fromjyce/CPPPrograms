#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

void method_one(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size(); 
    vector<vector<int>> C(n, vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void method_two(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size(); 
    vector<vector<int>> C(n, vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void method_three(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size(); 
    vector<vector<int>> C(n, vector<int>(n, 0));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void method_four(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size(); 
    vector<vector<int>> C(n, vector<int>(n, 0));
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < n; ++i) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void method_five(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size(); 
    vector<vector<int>> C(n, vector<int>(n, 0));
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void method_six(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size(); 
    vector<vector<int>> C(n, vector<int>(n, 0));
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


void fillMatrix(vector<vector<int>>& matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = rand() % 10; 
        }
    }
}

int main() {
    int n = 3000; 
    vector<vector<int>> A(n, vector<int>(n)), B(n, vector<int>(n)), C;

    fillMatrix(A, n);
    fillMatrix(B, n);

    auto start_one = high_resolution_clock::now();
    method_one(A, B);
    auto end_one = high_resolution_clock::now();
    auto duration_one = duration_cast<milliseconds>(end_one - start_one);
    cout << "Time taken for Method One: " << duration_one.count() << " milliseconds" << endl;
    
    auto start_two = high_resolution_clock::now();
    method_two(A, B);
    auto end_two = high_resolution_clock::now();
    auto duration_two = duration_cast<milliseconds>(end_two - start_two);
    cout << "Time taken for Method Two: " << duration_two.count() << " milliseconds" << endl;
    
    auto start_three = high_resolution_clock::now();
    method_three(A, B);
    auto end_three = high_resolution_clock::now();
    auto duration_three = duration_cast<milliseconds>(end_three - start_three);
    cout << "Time taken for Method Three: " << duration_three.count() << " milliseconds" << endl;
    
    auto start_four = high_resolution_clock::now();
    method_four(A, B);
    auto end_four = high_resolution_clock::now();
    auto duration_four = duration_cast<milliseconds>(end_four - start_four);
    cout << "Time taken for Method Four: " << duration_four.count() << " milliseconds" << endl;
    
    auto start_five = high_resolution_clock::now();
    method_five(A, B);
    auto end_five = high_resolution_clock::now();
    auto duration_five = duration_cast<milliseconds>(end_five - start_five);
    cout << "Time taken for Method Five: " << duration_five.count() << " milliseconds" << endl;
    
    auto start_six = high_resolution_clock::now();
    method_six(A, B);
    auto end_six = high_resolution_clock::now();
    auto duration_six = duration_cast<milliseconds>(end_six - start_six);
    cout << "Time taken for Method Six: " << duration_six.count() << " milliseconds" << endl;

    return 0;
}

