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

int m, n;
int sy, sx, ey, ex;
int visited[15][15], cnt, findAns;

void init() {
    findAns = 0;
}

void input() {
    cin >> m >> n;
    cin >> sy >> sx >> ey >> ex;
}

void f(int y, int x) {
    cnt++;
    visited[y][x] = 1;
    
    if (y == ey && x == ex && cnt == m * n) {
        findAns = 1;
        cout << 1 << '\n';
    }
    
    for (int i = 0; i < 4; i++) {
        if (findAns) break;
        
        int ny = y + dy[i], nx = x + dx[i];
        if (ny < 1 || m < ny || nx < 1 || n < nx) continue;
        if (visited[ny][nx]) continue;
        if (cnt < m * n - 1 && ny == ey && nx == ex) continue;
        
        f(ny, nx);
    }
    
    if (findAns) cout << y << ' ' << x << '\n';
    
    cnt--;
    visited[y][x] = 0;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int tc; cin >> tc;
	
	while (tc--) {
	    init();
	    
	    input();
	    
	    f(sy, sx);
	    
	    if (!findAns) cout << -1 << '\n';
	}
	
	return 0;
}
