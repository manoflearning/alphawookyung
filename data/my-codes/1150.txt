#include <bits/stdc++.h>
using namespace std;

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    int n, x = 0; cin >> n;
    for (int i = 0; i < n; i++) {
        int p; cin >> p;
        x ^= p;
    }

    cout << (x == 0 ? "cubelover" : "koosaga");

    return 0;
}