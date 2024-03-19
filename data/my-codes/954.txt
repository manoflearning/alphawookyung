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

struct yx {
	int y, x;
};

int n, m, dist[55][55];
string a[55];

void input() {
	cin >> n >> m;
	for (int i = 0; i < n; i++) cin >> a[i];
}

void init() {
	for (int i = 0; i < 55; i++) {
		for (int j = 0; j < 55; j++) {
			dist[i][j] = -1;
		}
	}
}

int bfs(int y, int x) {
	init();
	
	int ret = 0;
	
	queue<yx> q;
	
	q.push({ y, x });
	dist[y][x] = 0;

	while (!q.empty()) {
		auto v = q.front();
		q.pop();

		ret = max(ret, dist[v.y][v.x]);
		
		for (int i = 0; i < 4; i++) {
			int ny = v.y + dy[i], nx = v.x + dx[i];
			if (ny < 0 || n <= ny || nx < 0 || m <= nx) continue;
			if (dist[ny][nx] != -1) continue;
			if (a[ny][nx] == 'W') continue;
			
			dist[ny][nx] = dist[v.y][v.x] + 1;
			q.push({ ny, nx });
		}
	}

	return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	int ans = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (a[i][j] == 'L') {
				ans = max(ans, bfs(i, j));
			}
		}
	}

	cout << ans;

	return 0;
}