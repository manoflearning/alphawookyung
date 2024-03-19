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

int dp[5005][5005];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	memset(dp, -1, sizeof(dp));

	dp[0][0] = 0;
	for (int i = 1; i <= 5000; i += 2) {
		dp[i][i] = 2 * i - 1;
		dp[i - 1][i + 1] = dp[i][i] + 1;
		dp[i][i + 2] = dp[i][i] + 2;
		dp[i + 1][i + 1] = dp[i][i] + 3;
	}

	int tc; cin >> tc;
	while (tc--) {
		int x, y;
		cin >> x >> y;

		if (dp[y][x] == -1) cout << "No Number\n";
		else cout << dp[y][x] << '\n';
	}

	return 0;
}