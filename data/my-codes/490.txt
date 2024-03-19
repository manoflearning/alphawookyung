#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

const int MAX = 100;
const int dy[] = { 0, 0, 0, 1, -1 };
const int dx[] = { 0, 1, -1, 0, 0 };

int R, C, dp[MAX + 5][MAX + 5][5];
int sy, sx, sd, ey, ex, ed;
char arr[MAX + 5][MAX + 5];

void input();
void bfs();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	bfs();

	cout << dp[ey][ex][ed] << '\n';

	return 0;
}

void input() {
	cin >> R >> C;

	for (int i = 1; i <= R; i++) {
		for (int l = 1; l <= C; l++) {
			cin >> arr[i][l];
		}
	}

	cin >> sy >> sx >> sd;
	cin >> ey >> ex >> ed;
}

void bfs() {
	memset(dp, -1, sizeof(dp));

	struct yxd {
		int y, x, d;
	};

	queue<yxd> q;
	q.push({ sy, sx, sd });
	dp[sy][sx][sd] = 0;

	while (!q.empty() && dp[ey][ex][ed] == -1) {
		int y = q.front().y, x = q.front().x, d = q.front().d;
		q.pop();

		for (int nd = 1; nd <= 4; nd++) {
			if ((d <= 2 && nd <= 2) || (2 < d && 2 < nd)) continue;

			if (dp[y][x][nd] == -1) {
				dp[y][x][nd] = dp[y][x][d] + 1;
				q.push({ y, x, nd });
			}
		}

		for (int i = 1; i <= 3; i++) {
			int ny = y + i * dy[d], nx = x + i * dx[d];

			if (!(1 <= ny && ny <= R && 1 <= nx && nx <= C)) break;
			if (arr[ny][nx] == '1') break;

			if (dp[ny][nx][d] == -1) {
				dp[ny][nx][d] = dp[y][x][d] + 1;
				q.push({ ny, nx, d });
			}
		}
	}
}