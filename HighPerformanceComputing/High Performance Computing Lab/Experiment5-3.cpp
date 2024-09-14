#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>

#define MAX_SIZE 500

void matrix_multiply(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, std::vector<std::vector<int>> &C, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    std::ofstream outfile("matrix_times.csv");
    outfile << "Matrix Size,Time (milliseconds)\n";

    for (int size = 100; size <= MAX_SIZE; size += 100) {
        std::vector<std::vector<int>> A(size, std::vector<int>(size, 1));
        std::vector<std::vector<int>> B(size, std::vector<int>(size, 1));
        std::vector<std::vector<int>> C(size, std::vector<int>(size, 0));

        double start_time = omp_get_wtime();
        matrix_multiply(A, B, C, size);
        double end_time = omp_get_wtime();
        outfile << size << "," << (end_time - start_time)*1000 << "\n";
    }

    outfile.close();
    std::cout << "Execution times written to matrix_times.csv" << std::endl;
    return 0;
}
