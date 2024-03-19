#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

const int MAX = 30000, MAXD = 15;

int N, M, Q[5005];
vector<int> adj[MAX + 5];
int depth[MAX + 5], par[MAX + 5][MAXD + 5];

void input();
void dfs(int now, int prv);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	dfs(1, 0);

	for (int i = 1; i <= MAXD; i++) {
		for (int v = 1; v <= N; v++) {
			par[v][i] = par[par[v][i - 1]][i - 1];
		}
	}

	int ans = 0;
	
	for (int q = 1; q <= M - 1; q++) {
		int u = Q[q], v = Q[q + 1];
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
			su = par[su][0];
		}
		
		ans += depth[u] - depth[su] + depth[v] - depth[su];
	}

	cout << ans << '\n';

	return 0;
}

void input() {
	cin >> N;
	for (int i = 0; i < N - 1; i++) {
		int u, v;
		cin >> u >> v;
		adj[u].push_back(v);
		adj[v].push_back(u);
	}
	cin >> M;
	for (int i = 1; i <= M; i++) cin >> Q[i];
}

void dfs(int now, int prv) {
	for (int next : adj[now]) {
		if (next == prv) continue;
		depth[next] = depth[now] + 1;
		par[next][0] = now;
		dfs(next, now);
	}
}