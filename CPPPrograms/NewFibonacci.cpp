#include <iostream>
#include <vector>

using namespace std;
const int MOD = 1e9 + 7;
const int MAX_N = 1e7 + 1;

vector<int> zero_count(MAX_N), one_count(MAX_N);

void precompute() {
    zero_count[1] = 1; one_count[1] = 0;  // F(1) = "0"
    zero_count[2] = 0; one_count[2] = 1;  // F(2) = "1"
    
    for (int i = 3; i < MAX_N; i++) {
        zero_count[i] = (zero_count[i - 2] + zero_count[i - 1]) % MOD;
        one_count[i] = (one_count[i - 2] + one_count[i - 1]) % MOD;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    precompute();

    int Q, n;
    cin >> Q;
    while (Q--) {
        cin >> n;
        cout << abs(zero_count[n] - one_count[n]) << "\n";
    }

    return 0;
}
