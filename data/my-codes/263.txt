#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

const int MAX = 40000, MAXD = 15;

struct dv {
	int d, v;
};

vector<dv> adj[MAX + 5];
int depth[MAX + 5], dist[MAX + 5], par[MAX + 5][MAXD + 5];

void dfs(int now, int prv);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;
	for (int i = 0; i < N - 1; i++) {
		int u, v, d;
		cin >> u >> v >> d;
		adj[u].push_back({ d, v });
		adj[v].push_back({ d, u });
	}

	dfs(1, 0);

	for (int i = 1; i <= MAXD; i++) {
		for (int v = 1; v <= N; v++) {
			par[v][i] = par[par[v][i - 1]][i - 1];
		}
	}

	int Q; cin >> Q;
	while (Q--) {
		int u, v;
		cin >> u >> v;

		if (depth[u] > depth[v]) swap(u, v);

		int su = u, sv = v;
		
		int diff = depth[sv] - depth[su];
		for (int i = MAXD; i >= 0; i--) {
			if (diff & (1 << i)) {
				sv = par[sv][i];
				diff -= (1 << i);
			}
		}
		
		if (su != sv) {
			for (int i = MAXD; i >= 0; i--) {
				if (par[su][i] != par[sv][i]) {
					su = par[su][i];
					sv = par[sv][i];
				}
			}
		}
		
		int lca = su == sv ? su : par[su][0];

		int ans = dist[u] - dist[lca] + dist[v] - dist[lca];
		cout << ans << '\n';
	}

	return 0;
}

void dfs(int now, int prv) {
	depth[now] = depth[prv] + 1;
	par[now][0] = prv;
	for (auto e : adj[now]) {
		int next = e.v;
		if (next == prv) continue;
		dist[next] = dist[now] + e.d;
		dfs(next, now);
	}
}