#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

#define INTERVAL 10000

using namespace std;

int main() {
    int squarePoints = 0, circlePoints = 0;
    double estimatedPi, randX, randY, distance;
    srand(time(NULL));

    ofstream fobj;
    fobj.open("data.txt");
    
    clock_t start = clock();

    for (int i = 0; i < INTERVAL * INTERVAL; i++) {
        randX = double(rand() % (INTERVAL + 1)) / INTERVAL;
        randY = double(rand() % (INTERVAL + 1)) / INTERVAL;
        distance = (randX * randX) + (randY * randY);

        if (distance < 1) {
            circlePoints++;
        }
        squarePoints++;

        estimatedPi = double(4 * circlePoints) / squarePoints;
        if ((i + 1) % INTERVAL == 0) {
            double error = abs(M_PI - estimatedPi);
            double elapsedTime = double(clock() - start) / CLOCKS_PER_SEC;
            fobj << (i + 1) << " " << elapsedTime << " " << error << endl;
        }
    }
    fobj.close();

    cout << "The final estimated value is: " << estimatedPi << endl;
    cout << "The metrices are stored in data.txt" << endl;
    return 0;
}   