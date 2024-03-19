#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

char arr[55][55];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		for (int y = 1; y <= N; y++) {
			for (int x = 1; x <= N; x++) {
				cin >> arr[y][x];
			}
		}

		bool YES = true;
		for (int y = 1; y <= N; y++) {
			for (int x = 1; x <= N; x++) {
				if (arr[y][x] == '1') {
					if (y == N || arr[y + 1][x] == '1') continue;
					else if (x == N || arr[y][x + 1] == '1') continue;
					else YES = false;
				}
			}
		}

		if (YES) cout << "YES\n";
		else cout << "NO\n";
	}

	return 0;
}