#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

const int MAXN = 1005;

int r, c, dist[MAXN][MAXN], d, ey, ex;
char a[MAXN][MAXN];

void init() {
    d = -1;
    for (int i = 0; i < MAXN; i++) {
	    for (int j = 0; j < MAXN; j++) {
	        dist[i][j] = -1;
	    }
	}
}

void input() {
    cin >> c >> r;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            cin >> a[i][j];
        }
    }
}

void dfs(int y, int x, int h) {
    if (h > d) {
        d = h, ey = y, ex = x;
    }
    
    dist[y][x] = h;
    
    for (int i = 0; i < 4; i++) {
        int ny = y + dy[i], nx = x + dx[i];
        if (ny < 0 || r <= ny || nx < 0 || c <= nx) continue;
        if (a[ny][nx] == '#') continue;
        if (dist[ny][nx] != -1) continue;
        dfs(ny, nx, h + 1);
    }
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int tc; cin >> tc;
	
	while (tc--) {
	    input();
	    
	    init();
	    
	    for (int i = 0; i < r; i++) {
	        for (int j = 0; j < c; j++) {
	            if (a[i][j] == '#') continue;
	            
	            dfs(i, j, 0);
	            i = j = INF;
	            break;
	        }
	    }
	    
	    init();
	    
	    dfs(ey, ex, 0);
	    
	    assert(dist[ey][ex] >= 0);
	    cout << "Maximum rope length is " << dist[ey][ex] << ".\n";
	}
	
	return 0;
}
