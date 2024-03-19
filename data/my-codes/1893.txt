#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int N, K;
int arr[105][15];
bool visited[105][15], visited2[105][15];

int dfs1(int y, int x) {
	visited[y][x] = true;
	int ret = 1;
	for (int i = 0; i < 4; i++) {
		int ny = y + dy[i], nx = x + dx[i];
		if (ny < 1 || N < ny || nx < 1 || 10 < nx) continue;
		if (visited[ny][nx]) continue;
		if (arr[y][x] != arr[ny][nx]) continue;
		ret += dfs1(ny, nx);
	}
	return ret;
}

void dfs2(int y, int x) {
	visited2[y][x] = true;
	for (int i = 0; i < 4; i++) {
		int ny = y + dy[i], nx = x + dx[i];
		if (ny < 1 || N < ny || nx < 1 || 10 < nx) continue;
		if (visited2[ny][nx]) continue;
		if (arr[y][x] != arr[ny][nx]) continue;
		dfs2(ny, nx);
	}
	arr[y][x] = 0;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> K;
	for (int y = 1; y <= N; y++) {
		for (int x = 1; x <= 10; x++) {
			char c; cin >> c;
			arr[y][x] = c - '0';
		}
	}

	while (true) {
		memset(visited, 0, sizeof(visited));
		memset(visited2, 0, sizeof(visited2));

		bool exist = false;

		for (int y = 1; y <= N; y++) {
			for (int x = 1; x <= 10; x++) {
				if (visited[y][x] || arr[y][x] == 0) continue;
				if (dfs1(y, x) >= K) {
					dfs2(y, x);
					exist = true;
				}
			}
		}
		
		if (!exist) break;

		for (int y = N; y >= 1; y--) {
			for (int x = 1; x <= 10; x++) {
				if (arr[y][x] == 0) continue;
				for (int ny = y + 1; ny <= N; ny++) {
					if (arr[ny][x] != 0) break;
					swap(arr[ny][x], arr[ny - 1][x]);
				}
			}
		}
	}

	for (int y = 1; y <= N; y++) {
		for (int x = 1; x <= 10; x++) {
			cout << arr[y][x];
		}
		cout << '\n';
	}

	return 0;
}