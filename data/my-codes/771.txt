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

int N;
string a[16];
vector<int> adj[16];
int dp[1 << 16][16];

int f(int bit, int idx) {
	int& ret = dp[bit][idx];
	if (ret != -1) return ret;
	ret = 0;
	for (int i : adj[idx]) {
		if (bit & (1 << i)) continue;
		ret = max(ret, f(bit | (1 << i), i));
	}
	return ret += a[idx].size();
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	memset(dp, -1, sizeof(dp));

	cin >> N;
	for (int i = 0; i < N; i++) cin >> a[i];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i == j || a[i].back() != a[j].front()) continue;
			adj[i].push_back(j);
		}
	}

	int res = 0;
	for (int i = 0; i < N; i++) 
		res = max(res, f(1 << i, i));
	cout << res;

	return 0;
}