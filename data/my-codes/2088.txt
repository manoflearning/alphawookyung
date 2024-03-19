#include <iostream>
#include <vector>
using namespace std;

const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

struct yx {
	int y, x;
};

int R, C, T;
int arr[55][55];
yx air1, air2;

void input();
int simul();
int spread(int y, int x, vector<vector<int>>& s);
void move();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	int ans = simul();
	cout << ans << '\n';

	return 0;
}

void input() {
	cin >> R >> C >> T;

	for (int y = 1; y <= R; y++) {
		for (int x = 1; x <= C; x++) {
			cin >> arr[y][x];

			if (arr[y][x] == -1) {
				if (air1.x == 0) air1 = { y, x };
				else air2 = { y, x };
			}
		}
	}
}

int simul() {
	for (int i = 0; i < T; i++) {
		vector<vector<int>> s(55, vector<int>(55));

		for (int y = 1; y <= R; y++) {
			for (int x = 1; x <= C; x++) {
				if (arr[y][x] > 0) arr[y][x] -= spread(y, x, s);
			}
		}

		for (int y = 1; y <= R; y++) {
			for (int x = 1; x <= C; x++) {
				if (arr[y][x] != -1) arr[y][x] += s[y][x];
			}
		}

		move();
	}

	int ret = 0;
	for (int y = 1; y <= R; y++) {
		for (int x = 1; x <= C; x++) {
			if (arr[y][x] != -1) ret += arr[y][x];
		}
	}
	return ret;
}

int spread(int y, int x, vector<vector<int>>& s) {
	int ret = 0;
	for (int i = 0; i < 4; i++) {
		int ny = y + dy[i], nx = x + dx[i];
		if (ny < 1 || R < ny || nx < 1 || C < nx) continue;
		if (arr[ny][nx] == -1) continue;
		s[ny][nx] += arr[y][x] / 5;
		ret += arr[y][x] / 5;
	}
	return ret;
}

void move() {
	for (int y = air1.y - 2; y >= 1; y--) {
		arr[y + 1][1] = arr[y][1];
		arr[y][1] = 0;
	}
	for (int x = 2; x <= C; x++) {
		arr[1][x - 1] = arr[1][x];
		arr[1][x] = 0;
	}
	for (int y = 2; y <= air1.y; y++) {
		arr[y - 1][C] = arr[y][C];
		arr[y][C] = 0;
	}
	for (int x = C - 1; x >= 2; x--) {
		arr[air1.y][x + 1] = arr[air1.y][x];
		arr[air1.y][x] = 0;
	}

	for (int y = air2.y + 2; y <= R; y++) {
		arr[y - 1][1] = arr[y][1];
		arr[y][1] = 0;
	}
	for (int x = 2; x <= C; x++) {
		arr[R][x - 1] = arr[R][x];
		arr[R][x] = 0;
	}
	for (int y = R - 1; y >= air2.y; y--) {
		arr[y + 1][C] = arr[y][C];
		arr[y][C] = 0;
	}
	for (int x = C - 1; x >= 2; x--) {
		arr[air2.y][x + 1] = arr[air2.y][x];
		arr[air2.y][x] = 0;
	}
}