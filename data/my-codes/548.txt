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
#define vs vector<string>
#define vvs vector<vector<string>>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

vs a(5);
map<vs, int> dp;
int visited[5][5];

void input() {
	for (int i = 0; i < 5; i++) cin >> a[i];
}

void dfs(int y, int x, vs& b) {
	visited[y][x] = 1;
	for (int i = 0; i < 4; i++) {
		int ny = y + dy[i], nx = x + dx[i];
		if (ny < 0 || 5 <= ny || nx < 0 || 5 <= nx) continue;
		if (b[ny][nx] == '.') continue;
		if (visited[ny][nx]) continue;
		dfs(ny, nx, b);
	}
}

void init() {
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			visited[i][j] = 0;
		}
	}
}

int isCom(vs b) {
	init();

	int cnt = 0;
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			if (b[i][j] == '*' && !visited[i][j]) {
				cnt++;
				dfs(i, j, b);
			}
		}
	}
	return (cnt == 1);
}

int bfs() {
	queue<vs> q;

	q.push(a);
	dp.insert({ a, 0 });

	while (!q.empty()) {
		vs v = q.front();
		int d = dp.find(v)->second;
		q.pop();

		if (isCom(v)) return d;

		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				if (v[i][j] != '*') continue;

				for (int k = 0; k < 4; k++) {
					int ny = i + dy[k], nx = j + dx[k];
					if (ny < 0 || 5 <= ny || nx < 0 || 5 <= nx) continue;
					if (v[ny][nx] == '*') continue;

					swap(v[i][j], v[ny][nx]);
					if (dp.find(v) == dp.end()) {
						dp.insert({ v, d + 1 });
						q.push(v);
					}
					swap(v[i][j], v[ny][nx]);
				}
			}
		}
	}
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();

	cout << bfs();

	return 0;
}