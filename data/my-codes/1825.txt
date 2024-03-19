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
const int MAXN = 1e4;

int N, K, s[MAXN + 5];
int dp[MAXN + 5];

int f(int idx) {
	int& ret = dp[idx];
	if (ret != -1) return ret;
	if (idx == 0) return ret = 0;

	int mx = s[idx];
	for (int i = idx; i >= max(1, idx - K + 1); i--) {
		mx = max(mx, s[i]);
		ret = max(ret, f(i - 1) + mx * (idx - i + 1));
	}
	return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	memset(dp, -1, sizeof(dp));

	cin >> N >> K;
	for (int i = 1; i <= N; i++) cin >> s[i];

	cout << f(N);

	return 0;
}