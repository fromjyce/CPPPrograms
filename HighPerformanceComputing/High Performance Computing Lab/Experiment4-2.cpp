#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

std::vector<double> convolve1D_serial(const std::vector<double>& input, const std::vector<double>& kernel) {
    int input_len = input.size();
    int kernel_len = kernel.size();
    int output_len = input_len + kernel_len - 1;

    std::vector<double> output(output_len, 0.0);

    for (int i = 0; i < input_len; ++i) {
        for (int j = 0; j < kernel_len; ++j) {
            output[i + j] += input[i] * kernel[j];
        }
    }

    return output;
}

std::vector<double> convolve1D_parallel_static(const std::vector<double>& input, const std::vector<double>& kernel) {
    int input_len = input.size();
    int kernel_len = kernel.size();
    int output_len = input_len + kernel_len - 1;

    std::vector<double> output(output_len, 0.0);

    #pragma omp parallel
    {
        std::vector<double> output_private(output_len, 0.0);

        #pragma omp for schedule(static)
        for (int i = 0; i < input_len; ++i) {
            for (int j = 0; j < kernel_len; ++j) {
                output_private[i + j] += input[i] * kernel[j];
            }
        }

        #pragma omp critical
        {
            for (int k = 0; k < output_len; ++k) {
                output[k] += output_private[k];
            }
        }
    }

    return output;
}

int main() {
    std::random_device rd;
    std::mt19937 generator(rd());

    int size = 10000;

    std::vector<double> input(size);

    std::uniform_int_distribution<int> distribution(1, 100); 

    for (int i = 0; i < size; ++i) {
        input[i] = (double)distribution(generator);
    }

    std::vector<double> kernel = {0.5, 1.0, 0.5};

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> result_serial = convolve1D_serial(input, kernel);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_serial = end - start;
    std::cout << "Serial version time: " << elapsed_serial.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    std::vector<double> result_parallel_static = convolve1D_parallel_static(input, kernel);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_parallel_static = end - start;
    std::cout << "Parallel version time: " << elapsed_parallel_static.count() << " seconds\n";

    
    return 0;
}

