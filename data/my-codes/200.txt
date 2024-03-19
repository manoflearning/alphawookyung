#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int N;
int ans;

void simul(vector<vector<int>> arr, int turn);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;
	
	vector<vector<int>> arr(N, vector<int>(N));
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			cin >> arr[y][x];
		}
	}

	simul(arr, 0);

	cout << ans << '\n';

	return 0;
}

void simul(vector<vector<int>> arr, int turn) {
	if (turn == 5) {
		for (int y = 0; y < N; y++) {
			for (int x = 0; x < N; x++) {
				ans = max(ans, arr[y][x]);
			}
		}
		return;
	}
	
	vector<vector<int>> narr = arr;
	vector<vector<bool>> visited(N, vector<bool>(N));
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			if (narr[y][x] == 0) continue;
			for (int ny = y - 1; ny >= 0; ny--) {
				if (visited[ny][x]) {
					swap(narr[ny + 1][x], narr[y][x]);
					break;
				}
				if (narr[ny][x] == narr[y][x]) {
					narr[ny][x] <<= 1;
					narr[y][x] = 0;
					visited[ny][x] = true;
					break;
				}
				if (narr[ny][x] != 0) {
					swap(narr[ny + 1][x], narr[y][x]);
					break;
				}
				if (ny == 0) {
					swap(narr[ny][x], narr[y][x]);
					break;
				}
			}
		}
	}

	simul(narr, turn + 1);

	narr = arr;
	for (int y = 0; y < N; y++)
		fill(visited[y].begin(), visited[y].end(), false);
	for (int y = N - 1; y >= 0; y--) {
		for (int x = 0; x < N; x++) {
			if (narr[y][x] == 0) continue;
			for (int ny = y + 1; ny < N; ny++) {
				if (visited[ny][x]) {
					swap(narr[ny - 1][x], narr[y][x]);
					break;
				}
				if (narr[ny][x] == narr[y][x]) {
					narr[ny][x] <<= 1;
					narr[y][x] = 0;
					visited[ny][x] = true;
					break;
				}
				if (narr[ny][x] != 0) {
					swap(narr[ny - 1][x], narr[y][x]);
					break;
				}
				if (ny == N - 1) {
					swap(narr[ny][x], narr[y][x]);
					break;
				}
			}
		}
	}

	simul(narr, turn + 1);

	narr = arr;
	for (int y = 0; y < N; y++)
		fill(visited[y].begin(), visited[y].end(), false);
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			if (narr[y][x] == 0) continue;
			for (int nx = x - 1; nx >= 0; nx--) {
				if (visited[y][nx]) {
					swap(narr[y][nx + 1], narr[y][x]);
					break;
				}
				if (narr[y][nx] == narr[y][x]) {
					narr[y][nx] <<= 1;
					narr[y][x] = 0;
					visited[y][nx] = true;
					break;
				}
				if (narr[y][nx] != 0) {
					swap(narr[y][nx + 1], narr[y][x]);
					break;
				}
				if (nx == 0) {
					swap(narr[y][nx], narr[y][x]);
					break;
				}
			}
		}
	}

	simul(narr, turn + 1);

	narr = arr;
	for (int y = 0; y < N; y++)
		fill(visited[y].begin(), visited[y].end(), false);
	for (int y = 0; y < N; y++) {
		for (int x = N - 1; x >= 0; x--) {
			if (narr[y][x] == 0) continue;
			for (int nx = x + 1; nx < N; nx++) {
				if (visited[y][nx]) {
					swap(narr[y][nx - 1], narr[y][x]);
					break;
				}
				if (narr[y][nx] == narr[y][x]) {
					narr[y][nx] <<= 1;
					narr[y][x] = 0;
					visited[y][nx] = true;
					break;
				}
				if (narr[y][nx] != 0) {
					swap(narr[y][nx - 1], narr[y][x]);
					break;
				}
				if (nx == N - 1) {
					swap(narr[y][nx], narr[y][x]);
					break;
				}
			}
		}
	}

	simul(narr, turn + 1);
}