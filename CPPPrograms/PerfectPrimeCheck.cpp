#include <iostream>
#include <unordered_set>
#include <cmath>

using namespace std;

// Function to check if a number is prime
bool isPrime(int n) {
    if (n < 2) return false;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return false;
    }
    return true;
}

// Precompute squares of prime numbers
unordered_set<int> computePerfectSquares() {
    unordered_set<int> squares;
    for (int i = 2; i <= 100; i++) { // Up to sqrt(10^5)
        if (isPrime(i)) {
            squares.insert(i * i);
        }
    }
    return squares;
}

int main() {
    string word;
    cin >> word;

    unordered_set<int> validSquares = computePerfectSquares();

    int vowelCount = 0, consonantCount = 0;
    string vowels = "AEIOUaeiou";

    for (char ch : word) {
        if (vowels.find(ch) != string::npos)
            vowelCount++;
        else
            consonantCount++;
    }

    if (vowelCount >= 2 && consonantCount > 0 && validSquares.count(consonantCount)) {
        cout << "Qualify" << endl;
    } else {
        cout << "Disqualify" << endl;
    }

    return 0;
}
