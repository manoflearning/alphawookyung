#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
using namespace std;

const int MAXV = 1e5;

vector<int> adj[MAXV + 5], g[MAXV + 5];
int par[MAXV + 5], sz[MAXV + 5], dep[MAXV + 5];
int top[MAXV + 5], in[MAXV + 5], revIn[MAXV + 5], out[MAXV + 5], pv;
set<int> c[MAXV + 5];

void dfs(int v, int prv);
int dfs1(int v);
void dfs2(int v);
void modify(int v);
int query(int v);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;
	for (int i = 0; i < N - 1; i++) {
		int u, v;
		cin >> u >> v;
		adj[u].push_back(v);
		adj[v].push_back(u);
	}

	par[1] = 0;
	dfs(1, 0);
	dfs1(1);
	top[1] = 1; 
	dfs2(1);

	int M; cin >> M;
	while (M--) {
		int op, v;
		cin >> op >> v;
		if (op == 1) modify(v);
		else cout << query(v) << '\n';
	}

	return 0;
}

void dfs(int v, int prv) {
	for (auto& i : adj[v]) {
		if (i == prv) continue;
		g[v].push_back(i);
		dfs(i, v);
	}
}

int dfs1(int v) {
	sz[v] = 1;
	for (auto& i : g[v]) {
		par[i] = v, dep[i] = dep[v] + 1;
		sz[v] += dfs1(i);
		if (sz[i] > sz[g[v][0]]) swap(i, g[v][0]);
	}
	return sz[v];
}

void dfs2(int v) {
	in[v] = ++pv;
	revIn[in[v]] = v;
	for (auto& i : g[v]) {
		top[i] = (i == g[v][0] ? top[v] : i);
		dfs2(i);
	}
	out[v] = pv;
}

void modify(int v) {
	set<int>& s = c[top[v]];
	if (s.find(in[v]) != s.end()) s.erase(in[v]);
	else s.insert(in[v]);
}

int query(int v) {
	int ret = -1;
	while (v != 0) {
		int val = (c[top[v]].empty() ? -1 : *c[top[v]].begin());
		if (val != -1 && val <= in[v]) ret = revIn[val];
		v = par[top[v]];
	}
	return ret;
}