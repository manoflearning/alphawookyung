#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int n, a[20][20];
int dp[16][1 << 16];

void init() {
    memset(dp, -1, sizeof(dp));
}

void input() {
    cin >> n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> a[i][j];
        }
    }
}

int f(int v, int visited) {
    int& ret = dp[v][visited];
    if (ret != -1) return ret;
    if (visited == (1 << n) - 1) {
        return ret = (a[v][0] ? a[v][0] : INF);
    }
    
    ret = INF;
    for (int u = 0; u < n; u++) {
        if ((1 << u) & visited) continue;
        if (a[v][u] == 0) continue;
        ret = min(ret, a[v][u] + f(u, visited | (1 << u)));
    }
    
    return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	init();
	
	input();
	
	cout << f(0, (1 << 0));
	
	return 0;
}
