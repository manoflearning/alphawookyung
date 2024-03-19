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
const int MAX = 2000;

int n, A[MAX + 5], B[MAX + 5];
int dp[MAX + 5][MAX + 5];

int f(int a, int b) {
    int& ret = dp[a][b];
    if (ret != -1) return ret;
    if (a == n || b == n) return ret = 0;
    
    ret = max(f(a + 1, b), f(a + 1, b + 1));
    if (A[a] > B[b]) ret = max(ret, f(a, b + 1) + B[b]);
    return ret;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    memset(dp, -1, sizeof(dp));
    
    cin >> n;
    for (int i = 0; i < n; i++) cin >> A[i];
    for (int i = 0; i < n; i++) cin >> B[i];
    
    cout << f(0, 0);
    
    return 0;
}
