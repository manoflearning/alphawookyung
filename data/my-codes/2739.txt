#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int a[20][20];

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 20; j++) {
            cin >> a[i][j];
        }
    }

    int res = 0;

    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 20; j++) {
            if (i <= 20 - 4) res = max(res, a[i][j] * a[i + 1][j] * a[i + 2][j] * a[i + 3][j]);
            if (j <= 20 - 4) res = max(res, a[i][j] * a[i][j + 1] * a[i][j + 2] * a[i][j + 3]);
            if (i <= 20 - 4 && j <= 20 - 4) res = max(res, a[i][j] * a[i + 1][j + 1] * a[i + 2][j + 2] * a[i + 3][j + 3]);
            if (i <= 20 - 4 && 0 <= j - 3) res = max(res, a[i][j] * a[i + 1][j - 1] * a[i + 2][j - 2] * a[i + 3][j - 3]);
        }
    }

    cout << res;

    return 0;
}