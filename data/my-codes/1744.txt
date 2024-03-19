#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

const int dy[] = { -2, -1, 1, 2, 2, 1, -1, -2 };
const int dx[] = { 1, 2, 2, 1, -1, -2, -2, -1 };

struct yx {
	int y, x;
};

int N, dist[305][305];
yx st, en;

int bfs();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		cin >> N;
		cin >> st.y >> st.x;
		cin >> en.y >> en.x;

		cout << bfs() << '\n';
	}

	return 0;
}

int bfs() {
	memset(dist, -1, sizeof(dist));

	queue<yx> q;
	q.push(st);
	dist[st.y][st.x] = 0;

	while (!q.empty()) {
		int y = q.front().y, x = q.front().x;
		q.pop();

		if (en.y == y && en.x == x) return dist[y][x];

		for (int i = 0; i < 8; i++) {
			int ny = y + dy[i], nx = x + dx[i];
			if (ny < 0 || N <= ny || nx < 0 || N <= nx) continue;
			if (dist[ny][nx] != -1) continue;
			dist[ny][nx] = dist[y][x] + 1;
			q.push({ ny, nx });
		}
	}
}