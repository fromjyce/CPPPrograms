#include <iostream>
#include <fstream>
#include <omp.h>

long long parallel_factorial(int n) {
    long long result = 1;
    #pragma omp parallel for reduction(*:result)
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main() {
    int max_n = 20;
    std::ofstream outfile("factorial_times.csv");
    outfile << "N,Time (milliseconds)\n";

    for (int n = 10; n <= max_n; n += 5) {
        double start_time = omp_get_wtime();
        long long result = parallel_factorial(n);
        double end_time = omp_get_wtime();
        outfile << n << "," << (end_time - start_time)*1000 << "\n";
    }

    outfile.close();
    std::cout << "Execution times written to factorial_times.csv" << std::endl;
    return 0;
}
