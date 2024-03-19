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

int N, M;
int a[55][55];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> M;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			char c; cin >> c;
			a[i][j] = c - '0';
		}
	}

	for (int len = min(N, M) - 1; len >= 0; len--) {
		for (int i = 0; i < N - len; i++) {
			for (int j = 0; j < M - len; j++) {
				if (a[i][j] != a[i][j + len]) continue;
				if (a[i][j] != a[i + len][j]) continue;
				if (a[i][j] != a[i + len][j + len]) continue;
				cout << (len + 1) * (len + 1);
				return 0;
			}
		}
	}

	return 0;
}