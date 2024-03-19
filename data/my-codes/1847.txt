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

int r, c, q, a[1005][1005];
ll psum[1005][1005];

void input() {
    cin >> r >> c >> q;
    for (int i = 1; i <= r; i++) {
        for (int j = 1; j <= c; j++) {
            cin >> a[i][j];
        }
    }
}

void f() {
    psum[1][1] = a[1][1];
    for (int i = 2; i <= r; i++) {
        psum[i][1] = a[i][1] + psum[i - 1][1];
    }
    for (int i = 2; i <= c; i++) {
        psum[1][i] = a[1][i] + psum[1][i - 1];
    }
    
    for (int i = 2; i <= r; i++) {
        for (int j = 2; j <= c; j++) {
            psum[i][j] = a[i][j] + psum[i - 1][j] + psum[i][j - 1] - psum[i - 1][j - 1];
        }
    }
}

int query(int sy, int sx, int ey, int ex) {
    return psum[ey][ex] - psum[ey][sx - 1] - psum[sy - 1][ex] + psum[sy - 1][sx - 1];
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	f();
	
	for (int i = 0; i < q; i++) {
	    int sy, sx, ey, ex;
	    cin >> sy >> sx >> ey >> ex;
	    cout << query(sy, sx, ey, ex) / ((ey - sy + 1) * (ex - sx + 1)) << '\n';
	}
	
	return 0;
}
