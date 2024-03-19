#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int R, C, x, y;
		cin >> R >> C >> x >> y;

		vector<vector<char>> arr(R, vector<char>(C));

		int cnt = 0;
		for (int y = 0; y < R; y++) {
			for (int x = 0; x < C; x++) {
				cin >> arr[y][x];
				if (arr[y][x] == '.') cnt++;
			}
		}

		if (2 * x <= y) {
			cout << x * cnt << '\n';
			continue;
		}

		int cost = 0;
		for (int r = 0; r < R; r++) {
			int len = 0;
			for (int c = 0; c < C; c++) {
				if (arr[r][c] == '.') len++;
				else {
					cost += y * (len / 2);
					if (len % 2) cost += x;
					len = 0;
				}
			}
			if (len) {
				cost += y * (len / 2);
				if (len % 2) cost += x;
			}
		}

		cout << cost << '\n';
	}

	return 0;
}
