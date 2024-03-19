#include <iostream>
#include <vector>
#include <queue>
#include <cstring>
using namespace std;

const int MAX = 100;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

struct yx {
	int y, x;
};

int R, C, arr[MAX + 5][MAX + 5];
bool visited[MAX + 5][MAX + 5];

void input();
void bfs();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	int ans = 0;

	while (true) {
		memset(visited, 0, sizeof(visited));

		bfs();

		bool isCheeze = false;

		vector<yx> p;
		for (int y = 1; y <= R; y++) {
			for (int x = 1; x <= C; x++) {
				if (!visited[y][x]) {
					isCheeze = true;

					int cnt = 0;
					for (int i = 0; i < 4; i++) {
						int ny = y + dy[i], nx = x + dx[i];
						if (visited[ny][nx]) cnt++;
					}

					if (cnt >= 2) p.push_back({ y, x });
				}
			}
		}

		if (!isCheeze) break;

		ans++;

		for (auto& i : p)
			arr[i.y][i.x] = 0;
	}

	cout << ans << '\n';

	return 0;
}

void input() {
	cin >> R >> C;

	for (int y = 1; y <= R; y++) {
		for (int x = 1; x <= C; x++) {
			cin >> arr[y][x];
		}
	}
}

void bfs() {
	queue<yx> q;
	
	for (int y = 1; y <= R; y++) {
		q.push({ y, 1 });
		q.push({ y, C });
		visited[y][1] = visited[y][C] = true;
	}
	for (int x = 2; x <= C - 1; x++) {
		q.push({ 1, x });
		q.push({ R, x });
		visited[1][x] = visited[R][x] = true;
	}

	while (!q.empty()) {
		int y = q.front().y, x = q.front().x;
		q.pop();

		for (int i = 0; i < 4; i++) {
			int ny = y + dy[i], nx = x + dx[i];
			if (ny < 1 || R < ny || nx < 1 || C < nx) continue;
			if (!visited[ny][nx] && arr[ny][nx] == 0) {
				visited[ny][nx] = true;
				q.push({ ny, nx });
			}
		}
	}
}