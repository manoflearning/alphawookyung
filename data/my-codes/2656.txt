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
#define fr first
#define sc second

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1, 1, 1, -1, -1 };
const int dx[] = { 1, -1, 0, 0, 1, -1, 1, -1 };

int n;
vl a, b;
ll dp[105][105 * 105];

void init() {
	a.clear();
	b.clear();
	for (int i = 0; i < 105; i++) {
		for (int j = 0; j < 105 * 105; j++) {
			dp[i][j] = -1;
		}
	}
}

void input() {
	cin >> n;

	a.resize(n);
	b.resize(n);

	for (auto& i : a) cin >> i;
	for (auto& i : b) cin >> i;
}

ll f(int x, ll asum, ll bsum) {
	ll& ret = dp[x][asum];
	if (ret != -1) return ret;
	if (x == n) return ret = 0;

	ret = f(x + 1, asum + a[x], bsum + b[x]) + asum * a[x] + bsum * b[x];
	ret = min(ret, f(x + 1, asum + b[x], bsum + a[x]) + asum * b[x] + bsum * a[x]);

	return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		init();

		input();

		ll ans = 2 * f(0, 0, 0);
		for (auto& i : a) ans += i * i * (n - 1);
		for (auto& i : b) ans += i * i * (n - 1);

		cout << ans << '\n';
	}

	return 0;
}
