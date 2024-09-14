#include <omp.h>
#include <iostream>
int main() {
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < 5; ++i) {
std::cout << "Iteration " << i << " from thread " << omp_get_thread_num() << std::endl;
}
}
return 0;
}