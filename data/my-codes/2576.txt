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
		int n, m, a, b;
		cin >> n >> m >> a >> b;

		if (n * a != m * b) {
			cout << "NO\n";
			continue;
		}

		vector<vector<int>> ans(55, vector<int>(55));

		cout << "YES\n";
		int l = 1;
		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= a; j++) {
				ans[i][l] = 1;
				l = l % m + 1;
			}
		}
		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= m; j++) cout << ans[i][j];
			cout << '\n';
		}
	}

	return 0;
}