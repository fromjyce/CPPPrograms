#include <omp.h>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;

int factorial(int n) {
    if (n == 0 || n == 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

int factorial_parallel(int n) {
    if (n == 0 || n == 1) {
        return 1;
    } else {
        int result;
        #pragma omp parallel //multiple threads creation
        {
            #pragma omp single //only one thread performs -> avoiding race conditions
            {
                result = n * factorial_parallel(n - 1);
            }
        }
        return result;
    }
}

int main() {
    ofstream metrics_file("fibo_metrics.csv");
    if (!metrics_file.is_open()) {
        cerr << "Failed to open metrics.csv for writing." << endl;
        return 1;
    }
    metrics_file << "n,Serial Time (s),Parallel Time (s),Speedup,Efficiency" << endl;
    for (int n = 5; n <= 20; ++n) {
        auto start_serial = chrono::high_resolution_clock::now();
        int serial_result = factorial(n);
        auto end_serial = chrono::high_resolution_clock::now();
        chrono::duration<double> serial_duration = end_serial - start_serial;
        auto start_parallel = chrono::high_resolution_clock::now();
        int parallel_result = factorial_parallel(n);
        auto end_parallel = chrono::high_resolution_clock::now();
        chrono::duration<double> parallel_duration = end_parallel - start_parallel;
        double speedup = serial_duration.count() / parallel_duration.count();
        double efficiency = speedup / omp_get_max_threads();
        //cout << "n: " << n << " | Serial Time: " << serial_duration.count() 
             //<< " | Parallel Time: " << parallel_duration.count() 
             //<< " | Speedup: " << speedup 
             //<< " | Efficiency: " << efficiency << endl;
        metrics_file << n << "," 
                     << serial_duration.count() << "," 
                     << parallel_duration.count() << "," 
                     << speedup << "," 
                     << efficiency << endl;
    }
    metrics_file.close();

    return 0;
}

