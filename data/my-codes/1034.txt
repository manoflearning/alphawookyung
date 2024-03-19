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

int dp[31][2];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int D, K;
	cin >> D >> K;

	dp[1][0] = 1, dp[2][1] = 1;
	for (int i = 3; i <= D; i++) {
		dp[i][0] = dp[i - 1][0] + dp[i - 2][0];
		dp[i][1] = dp[i - 1][1] + dp[i - 2][1];
	}

	for (int i = 1; ; i++) {
		if ((K - i * dp[D][0]) % dp[D][1] == 0) {
			cout << i << '\n' << (K - i * dp[D][0]) / dp[D][1];
			break;
		}
	}

	return 0;
}