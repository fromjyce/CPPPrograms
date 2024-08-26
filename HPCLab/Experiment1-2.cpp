#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
using namespace std;
using namespace std::chrono;

int** generateMatrix(int row, int col) {
    int** matrix = new int*[row];
    for (int i = 0; i < row; i++) {
        matrix[i] = new int[col];
        for (int j = 0; j < col; j++) {
            matrix[i][j] = (rand() % 10);
        }
    }
    return matrix;
}

void displayMatrix(int row, int col, int** matrix) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

void freeMatrix(int row, int** matrix) {
    for (int i = 0; i < row; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

int** multiply_1(int** matrix, int** vector, int size) {
    int** result = new int*[size];
    for (int i = 0; i < size; i++) {
        result[i] = new int[1];
        result[i][0] = 0;
        for (int j = 0; j < size; j++) {
            result[i][0] += matrix[i][j]* vector[j][0];
        }
    }
    return result;
}

int** vectorAddition(int** vector1, int** vector2, int size) {
    int** result = new int*[size];
    for (int i = 0; i < size; i++) {
        result[i] = new int[1];
        result[i][0] = vector1[i][0] + vector2[i][0];
    }
    return result;
}

int** dotProduct(int** vector, int scalar, int size) {
    int** result = new int*[size];
    for (int i = 0; i < size; i++) {
        result[i] = new int[1];
        result[i][0] = scalar * vector[i][0];
    }
    return result;
}

int** multiply_2(int** matrix, int** vector, int size) {
    int** result = new int*[size];
    for (int i = 0; i < size; i++) {
        result[i] = new int[1];
        result[i][0] = 0;
    }

    for (int i = 0; i < size; i++) {
        int** tempVector = new int*[size];
        for (int j = 0; j < size; j++) {
            tempVector[j] = new int[1];
            tempVector[j][0] = matrix[j][i];
        }
        int** scaledVector = dotProduct(tempVector, vector[i][0], size);
        int** newResult = vectorAddition(result, scaledVector, size);

        for (int j = 0; j < size; j++) {
            result[j][0] = newResult[j][0];
        }

        freeMatrix(size, tempVector);
        freeMatrix(size, scaledVector);
        freeMatrix(size, newResult);
    }

    return result;
}

void getMetrices(int N[], int numSizes) {
    ofstream outFile("metrics.txt");
    outFile << "N Method1_Time(ms) Method2_Time(ms) Method1_FLOPs Method2_FLOPs\n";
    
    for (int i = 0; i < numSizes; i++) {
        int n = N[i];
        int** A = generateMatrix(n, n);
        int** x = generateMatrix(n, 1);
        auto start1 = high_resolution_clock::now();
        int** b1 = multiply_1(A, x, n);
        auto stop1 = high_resolution_clock::now();
        auto duration1 = duration_cast<milliseconds>(stop1 - start1).count();
        auto start2 = high_resolution_clock::now();
        int** b2 = multiply_2(A, x, n);
        auto stop2 = high_resolution_clock::now();
        auto duration2 = duration_cast<milliseconds>(stop2 - start2).count();
        long long method1_flops = 2LL * n * n;
        long long method2_flops = 3LL * n * n;
        outFile << n << " " << duration1 << " " << duration2 << " " << method1_flops << " " << method2_flops << "\n";
        freeMatrix(n, A);
        freeMatrix(n, x);
        freeMatrix(n, b1);
        freeMatrix(n, b2);
    }

    outFile.close();
}

int main() {
    srand(time(NULL));
    int n;
    cout << "Enter size of the matrix and the vector: "; 
    cin >> n;

    int** A = generateMatrix(n, n);
    cout << "Generated Matrix: " << endl;
    displayMatrix(n, n, A);

    cout << "Generated Vector: " << endl;
    int** x = generateMatrix(n, 1);
    displayMatrix(n, 1, x);

    cout << "Displaying results for multiplication to test the functions: " << endl;
    int **b1 = multiply_1(A, x, n);
    cout << "Multiplication result (by method 1): " << endl;
    displayMatrix(n, 1, b1);

    int **b2 = multiply_2(A, x, n);
    cout << "Multiplication result (by method 2): " << endl;
    displayMatrix(n, 1, b2);

    freeMatrix(n, A);
    freeMatrix(n, x);
    freeMatrix(n, b1);
    freeMatrix(n, b2);


    cout << "Now generating matrices and measuring time taken and FLOPs for sizes 512, 1024, 2048, 4096 and 8192" << endl;
    int N[] = {128, 512, 1024, 2048, 4096, 8192}, numSizes = sizeof(N) / sizeof(N[0]);   
    getMetrices(N, numSizes);
    cout << "The metrices measured are saved in 'metrices.txt' file" << endl;
    return 0;
}   