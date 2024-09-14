#include <omp.h>
#include <iostream>
int main() {
int sum = 0;
#pragma omp parallel reduction(+:sum)
{
#pragma omp for
for (int i = 0; i < 100; ++i) {
sum += i;
}
}
std::cout << "Sum = " << sum << std::endl;
return 0;
}