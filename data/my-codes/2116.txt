#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
#define uint unsigned int

const int MAXV = 5 * 1e5;

struct Seg {
	struct ab {
		uint a, b;
	};

	int flag;
	vector<uint> t;
	vector<ab> lazy;

	void build(int N) {
		for (flag = 1; flag < N; flag *= 2);
		t.resize(2 * flag);
		lazy.resize(2 * flag, { 1, 0 });
	}
	void modify(int L, int R, uint a, uint b) {
		return modify(L, R, 1, 0, flag - 1, a, b);
	}
	void modify(int L, int R, int n, int nL, int nR, uint a, uint b) {
		propagate(n, nL, nR);

		if (R < nL || nR < L) return;
		if (L <= nL && nR <= R) {
			lazy[n].a *= a;
			lazy[n].b *= a;
			lazy[n].b += b;

			propagate(n, nL, nR);
			return;
		}

		int mid = (nL + nR) >> 1;
		modify(L, R, n << 1, nL, mid, a, b);
		modify(L, R, n << 1 | 1, mid + 1, nR, a, b);

		t[n] = t[n << 1] + t[n << 1 | 1];
	}
	uint query(int L, int R) {
		return query(L, R, 1, 0, flag - 1);
	}
	uint query(int L, int R, int n, int nL, int nR) {
		propagate(n, nL, nR);

		if (R < nL || nR < L) return 0;
		if (L <= nL && nR <= R) return t[n];

		int mid = (nL + nR) >> 1;
		return query(L, R, n << 1, nL, mid) + query(L, R, n << 1 | 1, mid + 1, nR);
	}
	void propagate(int n, int nL, int nR) {
		if (lazy[n].a == 1 && lazy[n].b == 0) return;
		
		if (n < flag) {
			lazy[n << 1].a *= lazy[n].a;
			lazy[n << 1].b *= lazy[n].a;
			lazy[n << 1].b += lazy[n].b;
			lazy[n << 1 | 1].a *= lazy[n].a;
			lazy[n << 1 | 1].b *= lazy[n].a;
			lazy[n << 1 | 1].b += lazy[n].b;
		}
		t[n] = lazy[n].a * t[n] + lazy[n].b * (uint)(nR - nL + 1);
		lazy[n].a = 1, lazy[n].b = 0;
	}
}seg;

vector<int> adj[MAXV + 5], g[MAXV + 5];
int sz[MAXV + 5], dep[MAXV + 5], par[MAXV + 5];
int top[MAXV + 5], in[MAXV + 5], out[MAXV + 5], pv = -1;

void dfs(int v, int prv);
int dfs1(int v);
void dfs2(int v);
void updateSub(int v, uint a, uint b);
void updatePath(int u, int v, uint a, uint b);
uint querySub(int v);
uint queryPath(int u, int v);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N, Q;
	cin >> N >> Q;
	
	for (int i = 0; i < N - 1; i++) {
		int u, v;
		cin >> u >> v;
		adj[u].push_back(v);
		adj[v].push_back(u);
	}

	dfs(1, 0);
	top[1] = 1;
	dfs1(1); dfs2(1);

	seg.build(N);

	while (Q--) {
		int op, X, Y, V;
		cin >> op;

		if (op == 1) {
			cin >> X >> V;
			updateSub(X, 1, V);
		}
		else if (op == 2) {
			cin >> X >> Y >> V;
			updatePath(X, Y, 1, V);
		}
		else if (op == 3) {
			cin >> X >> V;
			updateSub(X, V, 0);
		}
		else if (op == 4) {
			cin >> X >> Y >> V;
			updatePath(X, Y, V, 0);
		}
		else if (op == 5) {
			cin >> X;
			cout << querySub(X) << '\n';
		}
		else if (op == 6) {
			cin >> X >> Y;
			cout << queryPath(X, Y) << '\n';
		}
	}

	return 0;
}

void dfs(int v, int prv) {
	for (int& i : adj[v]) {
		if (i == prv) continue;
		g[v].push_back(i);
		dfs(i, v);
	}
}

int dfs1(int v) {
	sz[v] = 1;
	for (int& i : g[v]) {
		dep[i] = dep[v] + 1, par[i] = v;
		sz[v] += dfs1(i);
		if (sz[i] > sz[g[v][0]]) swap(i, g[v][0]);
	}
	return sz[v];
}

void dfs2(int v) {
	in[v] = ++pv;
	for (int& i : g[v]) {
		top[i] = (i == g[v][0] ? top[v] : i);
		dfs2(i);
	}
	out[v] = pv;
}

void updateSub(int v, uint a, uint b) {
	seg.modify(in[v], out[v], a, b);
}

void updatePath(int u, int v, uint a, uint b) {
	while (top[u] != top[v]) {
		if (dep[top[u]] < dep[top[v]]) swap(u, v);
		int st = top[u];
		seg.modify(in[st], in[u], a, b);
		u = par[st];
	}
	if (dep[u] > dep[v]) swap(u, v);
	seg.modify(in[u], in[v], a, b);
}

uint querySub(int v) {
	return seg.query(in[v], out[v]);
}

uint queryPath(int u, int v) {
	uint ret = 0;
	while (top[u] != top[v]) {
		if (dep[top[u]] < dep[top[v]]) swap(u, v);
		int st = top[u];
		ret += seg.query(in[st], in[u]);
		u = par[st];
	}
	if (dep[u] > dep[v]) swap(u, v);
	ret += seg.query(in[u], in[v]);
	return ret;
}