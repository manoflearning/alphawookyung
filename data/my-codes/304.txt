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

int N, adj[20][20];
int dp[1 << 15][15][10];

int f(int bit, int now, int p) {
    int& ret = dp[bit][now][p];
    if(ret != -1) return ret;
    
    ret = 1;
    for(int i=0; i<N; i++) {
        if(bit & (1 << i)) continue;
        if(p > adj[now][i]) continue;
        ret = max(ret, f(bit | (1 << i), i, adj[now][i]) + 1);
    }
    return ret;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    cin>>N;
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            char c; cin>>c;
            adj[i][j]=c-'0';
        }
    }
    
    memset(dp, -1, sizeof(dp));

    cout<<f(1<<0, 0, 0);

    return 0;
}