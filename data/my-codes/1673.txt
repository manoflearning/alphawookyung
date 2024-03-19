#include <iostream>
using namespace std;

const int dy[] = { -1, 0, 1, 0 };
const int dx[] = { 0, 1, 0, -1 };

struct loc {
	int y, x, d;
};

int R, C;
loc v;
bool isWall[55][55], visited[55][55];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> R >> C;
	cin >> v.y >> v.x >> v.d;

	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			cin >> isWall[y][x];
		}
	}

	while (true) {
		visited[v.y][v.x] = true;

		bool isFound = false;

		for (int i = 1; i <= 4; i++) {
			loc n = { v.y + dy[(v.d - i + 4) % 4], v.x + dx[(v.d - i + 4) % 4], (v.d - i + 4) % 4 };

			if (n.y < 0 || R <= n.y || n.x < 0 || C <= n.x) continue;
			if (isWall[n.y][n.x] || visited[n.y][n.x]) continue;

			isFound = true;
			v = n;
			break;
		}

		if (!isFound) {
			loc n = { v.y - dy[v.d], v.x - dx[v.d], v.d };
			
			if (n.y < 0 || R <= n.y || n.x < 0 || C <= n.x) break;
			if (isWall[n.y][n.x]) break;

			v = n;
		}
	}

	int ans = 0;
	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			if (visited[y][x]) ans++;
		}
	}

	cout << ans << '\n';

	return 0;
}