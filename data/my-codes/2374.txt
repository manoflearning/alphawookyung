#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const long double PI = 3.141592653589793238462643383279502884197;

int arr[305][305];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		int R, C;
		cin >> R >> C;

		for (int y = 1; y <= R; y++) {
			for (int x = 1; x <= C; x++) {
				cin >> arr[y][x];
			}
		}

		bool cannot = false;
		for (int y = 1; y <= R; y++) {
			for (int x = 1; x <= C; x++) {
				if ((y == 1 || y == R) && (x == 1 || x == C)) {
					if (arr[y][x] > 2) cannot = true;
				}
				else if ((y == 1 || y == R) || (x == 1 || x == C)) {
					if (arr[y][x] > 3) cannot = true;
				}
				else {
					if (arr[y][x] > 4) cannot = true;
				}
			}
		}

		if (cannot) {
			cout << "NO\n";
			continue;
		}

		cout << "YES\n";
		for (int y = 1; y <= R; y++) {
			for (int x = 1; x <= C; x++) {
				if ((y == 1 || y == R) && (x == 1 || x == C)) {
					cout << 2 << ' ';
				}
				else if ((y == 1 || y == R) || (x == 1 || x == C)) {
					cout << 3 << ' ';
				}
				else {
					cout << 4 << ' ';
				}
			}
			cout << '\n';
		}
	}

	return 0;
}