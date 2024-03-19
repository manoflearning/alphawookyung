#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int dy[] = { 0, 1, 1 };
const int dx[] = { 1, 1, 0 };

int N, arr[21][21];
int dp[21][21][3];

int f(int y, int x, int dir);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	memset(dp, -1, sizeof(dp));
	
	cin >> N;

	for (int y = 1; y <= N; y++) {
		for (int x = 1; x <= N; x++) {
			cin >> arr[y][x];
		}
	}

	cout << f(1, 2, 0) << '\n';

	return 0;
}

int f(int y, int x, int dir) {
	int& ret = dp[y][x][dir];
	if (ret != -1) return ret;
	if (y == N && x == N) return ret = 1;

	ret = 0;
	for (int i = max(dir - 1, 0); i <= min(dir + 1, 2); i++) {
		int ny = y + dy[i], nx = x + dx[i];
		if (ny < 1 || N < ny || nx < 1 || N < nx) continue;
		if (arr[ny][nx] == 1) continue;
		if (i == 1 && (arr[ny - 1][nx] == 1 || arr[ny][nx - 1] == 1)) continue;
		ret += f(ny, nx, i);
	}
	
	return ret;
}