#include <iostream>
#include <fstream>
#include <omp.h>

long long fibonacci(int n) {
    if (n <= 1) return n;
    long long x, y;

    #pragma omp task shared(x)
    x = fibonacci(n - 1);

    #pragma omp task shared(y)
    y = fibonacci(n - 2);

    #pragma omp taskwait
    return x + y;
}

int main() {
    int max_n = 30;
    std::ofstream outfile("fibonacci_times.csv");
    outfile << "N,Time (milliseconds)\n";

    for (int n = 10; n <= max_n; n += 5) {
        double start_time = omp_get_wtime();
        long long result;
        #pragma omp parallel
        {
            #pragma omp single
            result = fibonacci(n);
        }
        double end_time = omp_get_wtime();
        outfile << n << "," << (end_time - start_time)*1000 << "\n";
    }

    outfile.close();
    std::cout << "Execution times written to fibonacci_times.csv" << std::endl;
    return 0;
}
