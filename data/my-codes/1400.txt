#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

const int MAX = 1e5, MAXD = 16;
const int INF = 1e9 + 7;

struct wv {
	int w, v;
};

int N, depth[MAX + 5], par[MAX + 5][MAXD + 5];
int pMin[MAX + 5][MAXD + 5], pMax[MAX + 5][MAXD + 5];
vector<wv> adj[MAX + 5];

void dfs(int now, int prv);
int LCA(int u, int v);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;

	for (int i = 0; i < N - 1; i++) {
		int u, v, w;
		cin >> u >> v >> w;
		adj[u].push_back({ w, v });
		adj[v].push_back({ w, u });
	}

	dfs(1, 0);

	for (int i = 1; i <= MAXD; i++) {
		for (int v = 1; v <= N; v++) {
			par[v][i] = par[par[v][i - 1]][i - 1];
			pMin[v][i] = min(pMin[v][i - 1], pMin[par[v][i - 1]][i - 1]);
			pMax[v][i] = max(pMax[v][i - 1], pMax[par[v][i - 1]][i - 1]);
		}
	}

	int Q; cin >> Q;
	while (Q--) {
		int u, v;
		cin >> u >> v;

		int lca = LCA(u, v);

		int dMin = INF, dMax = -INF;

		int diffU = depth[u] - depth[lca], diffV = depth[v] - depth[lca];
		for (int i = MAXD; i >= 0; i--) {
			if (diffU & (1 << i)) {
				diffU -= (1 << i);
				dMin = min(dMin, pMin[u][i]);
				dMax = max(dMax, pMax[u][i]);
				u = par[u][i];
			}
			if (diffV & (1 << i)) {
				diffV -= (1 << i);
				dMin = min(dMin, pMin[v][i]);
				dMax = max(dMax, pMax[v][i]);
				v = par[v][i];
			}
		}
		
		cout << dMin << ' ' << dMax << '\n';
	}

	return 0;
}

void dfs(int now, int prv) {
	for (auto e : adj[now]) {
		int next = e.v;
		if (next == prv) continue;
		depth[next] = depth[now] + 1;
		par[next][0] = now;
		pMin[next][0] = pMax[next][0] = e.w;
		dfs(next, now);
	}
}

int LCA(int u, int v) {
	if (depth[u] > depth[v]) swap(u, v);

	int diff = depth[v] - depth[u];
	for (int i = MAXD; i >= 0; i--) {
		if (diff & (1 << i)) {
			diff -= (1 << i);
			v = par[v][i];
		}
	}

	if (u != v) {
		for (int i = MAXD; i >= 0; i--) {
			if (par[u][i] != par[v][i]) {
				u = par[u][i];
				v = par[v][i];
			}
		}
		u = par[u][0];
		v = par[v][0];
	}

	return u;
}