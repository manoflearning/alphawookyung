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

int N, M, cnt;
char arr[25][25];
int dist[25][25];

int bfs(int y1, int x1, int y2, int x2) {
	for (int y = 0; y < N; y++)
		fill(dist[y], dist[y] + M, -1);

	queue<pii> q;
	q.push({ y1, x1 });
	q.push({ y2, x2 });
	dist[y1][x1] = dist[y2][x2] = 0;

	int nCnt = 0;

	while (!q.empty()) {
		int y = q.front().first, x = q.front().second;
		q.pop();

		if (arr[y][x] == '1') nCnt++;
		if (nCnt == cnt) return dist[y][x];

		for (int i = 0; i < 4; i++) {
			int ny = y + dy[i], nx = x + dx[i];
			if (ny < 0 || N <= ny || nx < 0 || M <= nx) continue;
			if (dist[ny][nx] != -1) continue;
			dist[ny][nx] = dist[y][x] + 1;
			q.push({ ny, nx });
		}
	}
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> M;
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < M; x++) {
			cin >> arr[y][x];
			if (arr[y][x] == '1') cnt++;
		}
	}

	int res = INF;
	for (int y1 = 0; y1 < N; y1++) {
		for (int x1 = 0; x1 < M; x1++) {
			for (int y2 = 0; y2 < N; y2++) {
				for (int x2 = 0; x2 < M; x2++) {
					if (arr[y1][x1] == '1' || arr[y2][x2] == '1') continue;
					if (y1 == y2 && x1 == x2) continue;
					res = min(res, bfs(y1, x1, y2, x2));
				}
			}
		}
	}

	cout << res;

	return 0;
}