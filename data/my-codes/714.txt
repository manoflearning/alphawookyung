#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
using namespace std;

const int MAXV = 1e5;
const int INF = 1e9 + 7;

int N, par[MAXV + 5][17], dep[MAXV + 5];
vector<int> adj[MAXV + 5];
int sz[MAXV + 5], cdtree[MAXV + 5];
bool used[MAXV + 5], color[MAXV + 5];
multiset<int> s[MAXV + 5];

void dfs(int now, int prv);
void cd(int now, int prv);
int getSize(int now, int prv);
int getCent(int now, int prv, int cnt);
void update(int v);
int query(int v);
int getDist(int u, int v);
int lca(int u, int v);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;
	for (int i = 0; i < N - 1; i++) {
		int u, v;
		cin >> u >> v;
		adj[u].push_back(v);
		adj[v].push_back(u);
	}

	dfs(1, -1);
	for (int i = 1; i <= 16; i++) {
		for (int j = 1; j <= N; j++) {
			par[j][i] = par[par[j][i - 1]][i - 1];
		}
	}

	cd(1, -1);

	int M; cin >> M;
	while (M--) {
		int op, v;
		cin >> op >> v;
		if (op == 1) update(v);
		else cout << query(v) << '\n';
	}

	return 0;
}

void dfs(int now, int prv) {
	dep[now] = dep[prv] + 1;
	par[now][0] = prv;
	for (int i : adj[now]) {
		if (i == prv) continue;
		dfs(i, now);
	}
}

void cd(int now, int prv) {
	int cnt = getSize(now, prv);
	int cent = getCent(now, prv, cnt);
	cdtree[cent] = prv;
	used[cent] = true;
	for (int i : adj[cent])
		if (!used[i]) cd(i,cent);
}

int getSize(int now, int prv) {
	sz[now] = 1;
	for (int i : adj[now]) {
		if (used[i] || i == prv) continue;
		sz[now] += getSize(i, now);
	}
	return sz[now];
}

int getCent(int now, int prv, int cnt) {
	for (int i : adj[now]) {
		if (used[i] || i == prv) continue;
		if (sz[i] > cnt / 2) return getCent(i, now, cnt);
	}
	return now;
}

void update(int v) {
	color[v] = !color[v];
	for (int i = v; i != -1; i = cdtree[i]) {
		int dist = getDist(i, v);
		if (color[v]) s[i].insert(dist);
		else s[i].erase(s[i].find(dist));
	}
}

int query(int v) {
	int ret = INF;
	for (int i = v; i != -1; i = cdtree[i]) {
		int dist = getDist(i, v);
		if (!s[i].empty()) ret = min(ret, dist + *s[i].begin());
	}
	return (ret == INF ? -1 : ret);
}

int getDist(int u, int v) {
	return dep[u] + dep[v] - 2 * dep[lca(u, v)];
}

int lca(int u, int v) {
	if (dep[u] < dep[v]) swap(u, v);

	int diff = dep[u] - dep[v];
	for (int i = 16; i >= 0; i--)
		if (diff & (1 << i)) u = par[u][i];

	if (u == v) return u;

	for (int i = 16; i >= 0; i--)
		if (par[u][i] ^ par[v][i]) u = par[u][i], v = par[v][i];
	return par[u][0];
}