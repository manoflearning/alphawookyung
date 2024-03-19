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

int R, C;
char a[100][100];
int res[100][100];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	memset(res, -1, sizeof(res));

	cin >> R >> C;
	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			cin >> a[y][x];
		}
	}

	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			if (a[y][x] == 'c') {
				for (int i = x; i < C; i++) {
					if (res[y][i] == -1) res[y][i] = i - x;
					else res[y][i] = min(res[y][i], i - x);
				}
			}
		}
	}

	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			cout << res[y][x] << ' ';
		}
		cout << '\n';
	}

	return 0;
}