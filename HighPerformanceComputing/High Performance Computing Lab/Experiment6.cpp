#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include <sstream>
#include <cstdlib>


void readSparseMatrix(const std::string& filename, std::vector<int>& row_idx, std::vector<int>& col_idx, std::vector<double>& values, int& num_rows, int& num_cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    std::istringstream ss(line);
    int nnz;
    ss >> num_rows >> num_cols >> nnz;
    for (int i = 0; i < nnz; i++) {
        int r, c;
        double v;
        file >> r >> c >> v;
        row_idx.push_back(r - 1);
        col_idx.push_back(c - 1);
        values.push_back(v);
    }

    file.close();
}

std::vector<double> sparseMatrixVectorMultiply(const std::vector<int>& row_idx, const std::vector<int>& col_idx, const std::vector<double>& values, const std::vector<double>& vec, int num_rows) {
    std::vector<double> result(num_rows, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < row_idx.size(); i++) {
        #pragma omp atomic
        result[row_idx[i]] += values[i] * vec[col_idx[i]];
    }

    return result;
}

int main() {
    std::string filename = "662_bus.mtx";
    std::vector<int> row_idx, col_idx;
    std::vector<double> values;
    int num_rows, num_cols;
    readSparseMatrix(filename, row_idx, col_idx, values, num_rows, num_cols);
    std::vector<double> vec(num_cols, 1.0);
    std::ofstream outfile("sparse_matrix_times.csv");
    outfile << "Num_Threads,Time (milliseconds)\n";

    for (int num_threads = 1; num_threads <= 8; num_threads *= 2) {
        omp_set_num_threads(num_threads);

        double start_time = omp_get_wtime();
        std::vector<double> result = sparseMatrixVectorMultiply(row_idx, col_idx, values, vec, num_rows);
        double end_time = omp_get_wtime();
        double time_ms = (end_time - start_time) * 1000;
        outfile << num_threads << "," << time_ms << "\n";
        std::cout << "Number of threads: " << num_threads << " - Time taken: " << time_ms << " ms" << std::endl;
    }

    outfile.close();
    return 0;
}
