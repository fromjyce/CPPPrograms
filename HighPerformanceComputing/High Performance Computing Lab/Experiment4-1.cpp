#include <iostream>
#include <omp.h>
#include <chrono>
#include <unistd.h>

void IAMLAZY(int i) {
    usleep(i * 1000);
}

int main() {
    auto start_static = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static, 2)
    for (int i = 1; i < 101; i++) {
        IAMLAZY(i);
    }
    auto end_static = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_static = end_static - start_static;
    std::cout << "Static Scheduling " << duration_static.count() << " seconds\n";
    
    
    auto start_dynamic = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 2)
    for (int i = 1; i < 101; i++) {
        IAMLAZY(i);
    }
    auto end_dynamic = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_dynamic = end_dynamic - start_dynamic;
    std::cout << "Dynamic Scheduling " << duration_dynamic.count() << " seconds\n";

    auto start_guided = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(guided, 2)
    for (int i = 1; i < 101; i++) {
        IAMLAZY(i);
    }
    auto end_guided = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_guided = end_guided - start_guided;
    std::cout << "Guided Scheduling " << duration_guided.count() << " seconds\n";

    return 0;
}

