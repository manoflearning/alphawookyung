#include <iostream>
#include <algorithm>
using namespace std;

const int dy[] = { 0, 0, 0, -1, 1 };
const int dx[] = { 0, 1, -1, 0, 0 };

struct dice {
	int y, x;
	int n[4][3];
};

int R, C, arr[25][25];
dice dc;

void roll(int mv);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int Q;
	cin >> R >> C >> dc.y >> dc.x >> Q;

	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			cin >> arr[y][x];
		}
	}

	while (Q--) {
		int mv; cin >> mv;

		int ny = dc.y + dy[mv], nx = dc.x + dx[mv];
		if (ny < 0 || R <= ny || nx < 0 || C <= nx) continue;

		roll(mv);
		dc.y = ny, dc.x = nx;

		if (arr[ny][nx] == 0) arr[ny][nx] = dc.n[1][1];
		else {
			dc.n[1][1] = arr[ny][nx];
			arr[ny][nx] = 0;
		}

		cout << dc.n[3][1] << '\n';
	}

	return 0;
}

void roll(int mv) {
	if (mv == 1) {
		for (int i = 0; i < 2; i++)
			swap(dc.n[1][i], dc.n[1][i + 1]);
		swap(dc.n[1][2], dc.n[3][1]);
	}
	else if (mv == 2) {
		swap(dc.n[1][2], dc.n[3][1]);
		for (int i = 1; i >= 0; i--)
			swap(dc.n[1][i], dc.n[1][i + 1]);
	}
	else if (mv == 3) {
		for (int i = 2; i >= 0; i--)
			swap(dc.n[i][1], dc.n[i + 1][1]);
	}
	else if (mv == 4) {
		for (int i = 0; i < 3; i++)
			swap(dc.n[i][1], dc.n[i + 1][1]);
	}
}