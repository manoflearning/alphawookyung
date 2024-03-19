#include <iostream>
#include <vector>
using namespace std;

const int dy[] = { 1, 0, -1, 0 };
const int dx[] = { 0, 1, 0, -1 };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N, flag; 
	cin >> N >> flag;

	vector<vector<int>> arr(N, vector<int>(N, -1));

	int y = -1, x = 0, dir = 0, ay = 0, ax = 0;
	for (int i = N * N; i >= 1; i--) {
		int ny = y + dy[dir], nx = x + dx[dir];
		if (ny < 0 || N <= ny || nx < 0 || N <= nx || arr[ny][nx] != -1) {
			dir = (dir + 1) % 4;
			ny = y + dy[dir], nx = x + dx[dir];
		}
		arr[ny][nx] = i;

		y = ny, x = nx;

		if (i == flag) ay = ny, ax = nx;
	}

	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			cout << arr[y][x] << ' ';
		}
		cout << '\n';
	}
	cout << ay + 1 << ' ' << ax + 1 << '\n';

	return 0;
}