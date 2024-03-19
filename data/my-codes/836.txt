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
const int MAXN = 1e5;

int n, a[MAXN + 5];
int adj[5][5];
int dp[5][5][MAXN + 5];

int f(int b1, int b2, int x) {
    int& ret = dp[b1][b2][x];
    if(ret != -1) return ret;
    if(x == n) return ret = 0;

    ret = INF;
    ret = min(ret, f(min(a[x], b2), max(a[x], b2), x + 1) + adj[b1][a[x]]);
    ret = min(ret, f(min(a[x], b1), max(a[x], b1), x + 1) + adj[b2][a[x]]);
    return ret;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    for(int i = 1; i <= 4; i++) {
        adj[i][i] = 1;
        adj[0][i] = 2;
        adj[i][i == 4 ? 1 : i + 1] = 3;
        adj[i == 4 ? 1 : i + 1][i] = 3;
        adj[i][i <= 2 ? i + 2 : i - 2] = 4;
    }

    for(int i = 0; ; i++) {
        cin>>a[i];
        if(a[i] == 0) {
            n = i;
            break;
        }
    }

    memset(dp, -1, sizeof(dp));
    
    cout<<f(0, 0, 0);
    
    return 0;
}