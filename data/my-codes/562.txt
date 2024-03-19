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
const int sz = 1e6;

ll n, dp[sz + 5];
int p, q, x, y;

ll f(ll n) {
    if (n <= 0) return 1;
    if (n <= sz) {
        ll& ret = dp[n];
        if (ret != -1) return ret;
        return ret = f(n / p - x) + f(n / q - y);
    }
    else return f(n / p - x) + f(n / q - y);
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    fill(dp, dp + sz + 1, -1);
    
    cin >> n >> p >> q >> x >> y;
    
    cout << f(n);
    
    return 0;
}
