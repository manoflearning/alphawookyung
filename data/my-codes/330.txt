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

int r, c, k, ans, d = 1;
int visited[5][5];
string a[5];

void input() {
    cin >> r >> c >> k;
    
    for (int i = 0; i < r; i++) cin >> a[i];
}

void f(int y, int x) {
    if (y == 0 && x == c - 1 && d == k) ans++;
    if (d == k) return;
    
    d++;
    visited[y][x] = 1;
    
    for (int i = 0; i < 4; i++) {
        int ny = y + dy[i], nx = x + dx[i];
        if (ny < 0 || r <= ny || nx < 0 || c <= nx) continue;
        if (a[ny][nx] == 'T') continue;
        if (visited[ny][nx]) continue;
        
        f(ny, nx);
    }
    
    d--;
    visited[y][x] = 0;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	f(r - 1, 0);
	
	cout << ans;
	
	return 0;
}
