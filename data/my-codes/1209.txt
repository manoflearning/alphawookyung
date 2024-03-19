#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const int MAXV = 1e5, MAXD = 16;

vector<int> adj[MAXV + 5];
int dep[MAXV + 5], par[MAXV + 5][MAXD + 5];
int isG[MAXV + 5], bit[MAXV + 5][MAXD + 5];

void dfs(int now, int prv);
bool query(int u, int v, int wantG);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int N, M;
	string S;
	cin >> N >> M >> S;
	for (int i = 0; i < S.size(); i++)
		isG[i + 1] = (S[i] == 'G');
	
	for (int i = 0; i < N - 1; i++) {
		int u, v;
		cin >> u >> v;
		adj[u].push_back(v);
		adj[v].push_back(u);
	}

	dfs(1, 0);

	for (int i = 1; i <= MAXD; i++) {
		for (int j = 1; j <= N; j++) {
			par[j][i] = par[par[j][i - 1]][i - 1];
			bit[j][i] = bit[j][i - 1] | bit[par[j][i - 1]][i - 1];
		}
	}
	
	string res;
	while (M--) {
		int u, v;
		char C;
		cin >> u >> v >> C;
		res.push_back(query(u, v, C == 'G') + '0');
	}

	cout << res;

	return 0;
}

void dfs(int now, int prv) {
	dep[now] = dep[prv] + 1;
	par[now][0] = prv;
	bit[now][0] = (1 << isG[now]);
	for (int i : adj[now])
		if (i != prv) dfs(i, now);
}

bool query(int u, int v, int wantG) {
	if (dep[u] < dep[v]) swap(u, v);

	int ret = 0;

	int diff = dep[u] - dep[v];
	for (int i = 0; i <= MAXD; i++) {
		if (diff & (1 << i)) {
			ret |= bit[u][i];
			u = par[u][i];
		}
	}

	if (u == v) return (ret | bit[u][0]) & (1 << wantG);
	
	for (int i = MAXD; i >= 0; i--) {
		if (par[u][i] ^ par[v][i]) {
			ret |= bit[u][i] | bit[v][i];
			u = par[u][i], v = par[v][i];
		}
	}

	return (ret | bit[u][1] | bit[v][0]) & (1 << wantG);
}