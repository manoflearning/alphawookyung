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

const int MAX = 2e3;

int n;
vector<int> adj[MAX + 5];
int dp[MAX + 5], sz[MAX + 5];

void init() {
	memset(dp, -1, sizeof(dp));
}

int dfs(int v, bool b) {
	int& ret = dp[v];
	if (ret != -1) return ret;
	
	ret = b, sz[v] = 1;

	for (int n : adj[v]) {
		if (dp[n] == -1) {
			ret += dfs(n, !b);
			sz[v] += sz[n];
		}
	}

	return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	init();

	cin >> n;

	for (int i = 0; i < n; i++) {
		string s; cin >> s;
		for (int j = 0; j < n; j++) {
			if (s[j] == '1') { 
				adj[i].push_back(j);
				adj[j].push_back(i);
			}
		}
	}

	int ans = 0;

	for (int i = 0; i < n; i++) {
		if (dp[i] == -1) {
			int sum = dfs(i, 0);
			ans += max(sum, sz[i] - sum);
		}
	}

	cout << ans;

	return 0;
}