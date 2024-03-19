#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>
#define vs vector<string>
#define vvs vector<vector<string>>

const int INF = 1e9 + 7;
const int MOD = 998244353;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int n, k, a[305], p[305], q[305];
vvi dp;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> n >> k;
	for (int i = 0; i < n; i++) cin >> p[i];
	for (int i = 0; i < n; i++) cin >> q[i];

	for (int i = 0; i < n; i++) a[p[i] - 1] = q[i] - 1;
	
	dp.resize(k + 1, vi(n + 1));

	dp[0][n] = 1;

	for (int i = 0; i < n; i++) {
		vvi ndp(k + 1, vi(n + 1));
		for (int c = 0; c <= k; c++) {
			for (int r = 0; r <= n; r++) {
				if (c < k && a[i] < r) {
					ndp[c + 1][r] += dp[c][r];
					ndp[c + 1][r] %= MOD;
				}
				ndp[c][min(r, a[i])] += dp[c][r];
				ndp[c][min(r, a[i])] %= MOD;
			}
		}
		dp = ndp;
	}

	int ans = 0;
	for (int i = 0; i <= n; i++) ans = (ans + dp[k][i]) % MOD;

	cout << ans;

	return 0;
}