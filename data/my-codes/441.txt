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
const int dy[] = { 0, 0, 1, -1, 1, 1, -1, -1 };
const int dx[] = { 1, -1, 0, 0, 1, -1, 1, -1 };

int n, m, a[105][75], visited[105][75];

void input() {
	cin >> n >> m;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			cin >> a[i][j];
		}
	}
}

void f2(int y, int x) {
	visited[y][x] = 1;
	for (int i = 0; i < 8; i++) {
		int ny = y + dy[i], nx = x + dx[i];
		if (ny < 0 || n <= ny || nx < 0 || m <= nx) continue;
		if (visited[ny][nx]) continue;
		if (a[ny][nx] > a[y][x]) continue;
		f2(ny, nx);
	}
}

int f() {
	int ret = 0;

	for (int i = 500; i >= 0; i--) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < m; k++) {
				if (a[j][k] != i) continue;
				if (visited[j][k]) continue;
				f2(j, k);
				ret++;
			}
		}
	}
	
	return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	cout << f();

	return 0;
}