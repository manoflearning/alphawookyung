#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const long double PI = 3.141592653589793238462643383279502884197;
const int MAXV = 2 * 1e5;

struct Seg {
	int flag;
	vector<ll> t;

	void build(int N) {
		for (flag = 1; flag < N; flag *= 2);
		t.resize(2 * flag);
		for (int i = flag - 1; i >= 1; i--) t[i] = t[i << 1] + t[i << 1 | 1];
	}
	void modify(int p, int value) {
		for (t[p += flag - 1] = value; p > 1; p >>= 1) t[p >> 1] = t[p] + t[p ^ 1];
	}
	int query(int L, int R) {
		return query(L, R, 1, 1, flag);
	}
	int query(int L, int R, int n, int nL, int nR) {
		if (R < nL || nR < L) return 0;
		if (L <= nL && nR <= R) return t[n];

		int mid = (nL + nR) >> 1;
		return query(L, R, n << 1, nL, mid) + query(L, R, n << 1 | 1, mid + 1, nR);
	}
}seg;

struct vc {
	int v, c;

	bool operator<(vc& rhs) {
		return c < rhs.c;
	}
};

vector<int> color[MAXV + 5];
vector<int> adj[MAXV + 5], g[MAXV + 5];
int sz[MAXV + 5], dep[MAXV + 5], par[MAXV + 5];
int top[MAXV + 5], in[MAXV + 5], out[MAXV + 5], pv;
vc Q[MAXV + 5];

void dfs(int v, int prv);
int dfs1(int v);
void dfs2(int v);
void modify(int v, ll value);
ll query(int u, int v);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N, M, C;
	cin >> N >> M >> C;

	for (int i = 1; i <= N; i++) {
		int x; cin >> x;
		color[x].push_back(i);
	}

	for (int i = 0; i < N - 1; i++) {
		int u, v;
		cin >> u >> v;
		adj[u].push_back(v);
		adj[v].push_back(u);
	}
	
	seg.build(N);

	dfs(1, 0);
	top[1] = 1;
	dfs1(1); dfs2(1);

	for (int i = 0; i < M; i++) {
		cin >> Q[i].v >> Q[i].c;
	}

	sort(Q, Q + M);

	ll ans = 0;
	
	for (int l = 1; l <= Q[0].c; l++) {
		for (auto k : color[l]) {
			seg.modify(in[k], 1);
		}
	}
	for (int i = 0; i < M; i++) {
		if (i != 0 && Q[i - 1].c < Q[i].c) {
			for (int l = Q[i - 1].c + 1; l <= Q[i].c; l++) {
				for (auto k : color[l]) seg.modify(in[k], 1);
			}
		}

		ans = (ans + seg.query(in[Q[i].v], out[Q[i].v])) % MOD;
	}

	cout << ans;
	
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
		dep[i] = dep[v] + 1, par[i] = v;
		sz[v] += dfs1(i);
		if (sz[i] > sz[g[v][0]]) swap(i, g[v][0]);
	}
	return sz[v];
}

void dfs2(int v) {
	in[v] = ++pv;
	for (auto& i : g[v]) {
		top[i] = (i == g[v][0] ? top[v] : i);
		dfs2(i);
	}
	out[v] = pv;
}

void modify(int v, ll value) {
	seg.modify(in[v], value);
}

ll query(int u, int v) {
	ll ret = 0;
	while (top[u] ^ top[v]) {
		if (dep[top[u]] < dep[top[v]]) swap(u, v);
		int st = top[u];
		ret += seg.query(in[st], in[u]);
		u = par[st];
	}
	if (dep[u] > dep[v]) swap(u, v);
	ret += seg.query(in[u], in[v]);
	return ret;
}