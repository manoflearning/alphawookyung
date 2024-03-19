#include <bits/stdc++.h>
using namespace std;

int n;
vector<int> b;
vector<vector<int>> a;

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    cin >> n;
    b.resize(n);
    for (auto& i : b) cin >> i;

    a.resize(n, vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = i * j % n;
        }
    }

    for (int i = 0; i < n; i++) {
        int diff = b[i] - a[i][i];
        for (int j = 0; j < n; j++) {
            a[i][j] = (a[i][j] + diff + n) % n;
            cout << a[i][j] << ' ';
        }
        cout << '\n';
    }

    return 0;
}