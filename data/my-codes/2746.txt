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

int f(ll x) {
    if (x == 1) return 1;
    
    if (x & 1) return f(3 * x + 1) + 1;
    else return f(x / 2) + 1;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    int mx = 0, res;
    for (int i = 1; i < 1e6; i++) {
        int val = f(i);
        if (mx < val) {
            mx = val, res = i;
        }
    }

    cout << res;

    return 0;
}