#include <iostream>
#include <cstring>
#include <algorithm>
#include <queue>
using namespace std;

const int dy[] = { 1, -1, 0, 0, -2, -1, 1, 2, 2, 1, -1, -2 };
const int dx[] = { 0, 0, 1, -1, 1, 2, 2, 1, -1, -2, -2, -1 };
const int INF = 1e9 + 7;

int K, R, C;
int arr[205][205];
bool visited[205][205][31];

int bfs();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> K >> C >> R;

	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			cin >> arr[y][x];
		}
	}

	cout << bfs();

	return 0;
}

int bfs() {
	struct yxcnt {
		int y, x, cnt, d;
	};

	queue<yxcnt> q;
	q.push({ 0, 0, 0, 0 });
	visited[0][0][0] = true;

	while (!q.empty()) {
		int y = q.front().y, x = q.front().x, cnt = q.front().cnt, d = q.front().d;
		q.pop();

		if (y == R - 1 && x == C - 1) return d;

		for (int i = 0; i < 4; i++) {
			int ny = y + dy[i], nx = x + dx[i];
			if (ny < 0 || R <= ny || nx < 0 || C <= nx) continue;
			if (visited[ny][nx][cnt] || arr[ny][nx] == 1) continue;

			visited[ny][nx][cnt] = true;
			q.push({ ny, nx, cnt, d + 1 });
		}
		if (cnt < K) {
			for (int i = 4; i < 12; i++) {
				int ny = y + dy[i], nx = x + dx[i];
				if (ny < 0 || R <= ny || nx < 0 || C <= nx) continue;
				if (visited[ny][nx][cnt + 1] || arr[ny][nx] == 1) continue;

				visited[ny][nx][cnt + 1] = true;
				q.push({ ny, nx, cnt + 1, d + 1 });
			}
		}
	}
	return -1;
}