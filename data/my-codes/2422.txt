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
const int MAXV = 1e5;

int n, m;
vector<ll> p, h, sz;
vector<int> adj[MAXV + 5];
bool YES;

void init(int n) {
	for (int i = 1; i <= n; i++) adj[i].clear();
}

ll dfs(int now, int prv) {
	sz[now] = p[now];
	for (int i : adj[now]) {
		if (i != prv) sz[now] += dfs(i, now);
	}
	return sz[now];
}

pll dfs2(int now, int prv) {
	pll res = { 0, 0 };
	for (int i : adj[now]) {
		if (i != prv) {
			pll j = dfs2(i, now);
			res.first += j.first;
			res.second += j.second;
		}
	}

	ll f = (sz[now] + h[now]) >> 1, s = (sz[now] - h[now]) >> 1;

	if (!(res.first + res.second == f + s - p[now] && res.second >= s - p[now])) YES = false;
	if (fabs(h[now]) > sz[now]) YES = false;

	return { f, s };
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;
	while (tc--) {
		cin >> n >> m;

		p.resize(n + 1), h.resize(n + 1);
		for (int i = 1; i <= n; i++) cin >> p[i];
		for (int i = 1; i <= n; i++) cin >> h[i];

		for (int i = 0; i < n - 1; i++) {
			int u, v;
			cin >> u >> v;
			adj[u].push_back(v);
			adj[v].push_back(u);
		}

		sz.resize(n + 1);
		dfs(1, 0);

		YES = true;
		dfs2(1, 0);

		if (YES) cout << "YES\n";
		else cout << "NO\n";

		init(n);
	}

	return 0;
}