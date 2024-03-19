#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const ll INF = (ll)1e18 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const int MAX = 1e4;

struct wvnum {
	int w, v, num;
};

struct vprv {
	int v, prv;
};

int n;
ll dp[2 * MAX + 5], dp2[2 * MAX + 5];
vector<wvnum> adj[MAX + 5];
vprv edge[2 * MAX + 5];

void init() {
	fill(dp, dp + 2 *  MAX + 5, -1);
	fill(dp2, dp2 + 2 * MAX + 5, -1);
	for (int i = 0; i < MAX + 5; i++) adj[i].clear();
}

ll f(int num);
ll f2(int num);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	while (1) {
		init();

		cin >> n;

		if (n == 0) break;

		for (int i = 0; i < n - 1; i++) {
			int u, v, w;
			cin >> u >> v >> w;
			
			adj[u].push_back({ w, v, i << 1 });
			adj[v].push_back({ w, u, i << 1 | 1 });
			edge[i << 1] = { v, u };
			edge[i << 1 | 1] = { u, v };
		}

		ll res = INF;
		for (int v = 0; v < n; v++) {
			ll sum = 0;
			for (auto& i : adj[v]) {
				sum += f(i.num) + i.w * f2(i.num);
			}
			res = min(res, sum);
		}

		cout << res << '\n';
	}

	return 0;
}

ll f(int num) {
	ll& ret = dp[num];
	if (ret != -1) return ret;

	int v = edge[num].v, prv = edge[num].prv;

	ret = 0;
	for (auto& i : adj[v]) {
		if (i.v == prv) continue;
		ret += f(i.num) + i.w * f2(i.num);
	}
	return ret;
}

ll f2(int num) {
	ll& ret = dp2[num];
	if (ret != -1) return ret;

	int v = edge[num].v, prv = edge[num].prv;

	ret = 1;
	for (auto& i : adj[v]) {
		if (i.v == prv) continue;
		ret += f2(i.num);
	}
	return ret;
}