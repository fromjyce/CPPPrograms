#include <omp.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
using namespace std;


void column_sum_serial(int** arr, int* b, int N) {
    for (int i = 0; i < N; ++i) {
        int sum = 0;
        for (int j = 0; j < N; ++j) 
            sum += arr[j][i];
        b[i] = sum;
    }
}


void column_sum_parallel(int** arr, int* b, int N) {
    #pragma omp parallel for collapse(2) //combines into a single parallel -> distributes work
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            #pragma omp atomic
            b[i] += arr[j][i]; //only one thread can modify -> race condition prevention
        }
    }
}

// Driver code
int main() {
    int sizes[] = {256, 512, 1024, 2048};
    int thread_counts[] = {1, 2, 4, 6};

    ofstream results("results.csv");
    results << "N,Threads,Serial Time,Parallel Time,Speedup,Efficiency\n";

    for (int size : sizes) {
        for (int threads : thread_counts) {
            int N = size;
            omp_set_num_threads(threads);
            int** arr = new int*[N];
            for (int i = 0; i < N; i++) {
                arr[i] = new int[N];
            }

            int* row_sum = new int[N];
            int x = 1;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    arr[i][j] = x++;
                }
            }
            auto start_serial = chrono::high_resolution_clock::now();
            column_sum_serial(arr, row_sum, N);
            auto end_serial = chrono::high_resolution_clock::now();
            double serial_time = chrono::duration<double>(end_serial - start_serial).count();
            auto start_parallel = chrono::high_resolution_clock::now();
            column_sum_parallel(arr, row_sum, N);
            auto end_parallel = chrono::high_resolution_clock::now();
            double parallel_time = chrono::duration<double>(end_parallel - start_parallel).count();


            double speedup = serial_time / parallel_time;
            double efficiency = speedup / threads;

            //cout << "N: " << size << ", Threads: " << threads << "\n";
            //cout << "Serial Time: " << serial_time << " seconds\n";
            //cout << "Parallel Time: " << parallel_time << " seconds\n";
            //cout << "Speedup: " << speedup << "\n";
            //cout << "Efficiency: " << efficiency << "\n";

            results << size << "," << threads << "," << serial_time << "," << parallel_time << "," << speedup << "," << efficiency << "\n";


            delete[] row_sum;
            for (int i = 0; i < N; i++) {
                delete[] arr[i];
            }
            delete[] arr;
        }
    }

    results.close();
    return 0;
}

