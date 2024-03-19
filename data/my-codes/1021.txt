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

ll dp[10][65][65];

ll f(int num, int idx, int n) {
	ll& ret = dp[num][idx][n];
	if (ret != -1) return ret;
	if (idx == n) return ret = 1;
	
	ret = 0;
	for (int i = num; i < 10; i++)
		ret += f(i, idx + 1, n);
	return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	memset(dp, -1, sizeof(dp));

	int tc; cin >> tc;
	while (tc--) {
		int n; cin >> n;
		cout << f(0, 0, n) << '\n';
	}

	return 0;
}