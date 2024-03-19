#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;

const int MAX = 750;
const int dy[] = { 1, 1, -1, -1 };
const int dx[] = { 1, -1, 1, -1 };

int R, C, arr[MAX + 5][MAX + 5];
int dp[MAX + 5][MAX + 5][4];

int f(int y, int x, int i);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> R >> C;

	for (int y = 1; y <= R; y++) {
		for (int x = 1; x <= C; x++) {
			char c; cin >> c;
			arr[y][x] = c - '0';
		}
	}

	memset(dp, -1, sizeof(dp));

	for (int i = 0; i < 4; i++) {
		for (int y = 1; y <= R; y++) {
			for (int x = 1; x <= C; x++) {
				f(y, x, i);
			}
		}
	}

	int upper = (min(R, C) + min(R, C) % 2) / 2;

	for (int s = upper; s >= 1; s--) {
		int len = s - 1;

		for (int y = s; y <= R - s + 1; y++) {
			for (int x = s; x <= C - s + 1; x++) {				
				if (dp[y - len][x][0] < s) continue;
				if (dp[y - len][x][1] < s) continue;
				if (dp[y + len][x][2] < s) continue;
				if (dp[y + len][x][3] < s) continue;

				cout << s;
				return 0;
			}
		}
	}

	cout << 0;

	return 0;
}

int f(int y, int x, int i) {
	int& ret = dp[y][x][i];
	if (ret != -1) return ret;
	if (arr[y][x] == 0) return ret = 0;

	return ret = f(y + dy[i], x + dx[i], i) + 1;
}