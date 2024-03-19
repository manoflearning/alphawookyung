#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

const int MAXN = 1005, MAXD = 10;

struct wv {
    int w, v;
};

int n, m, dp[MAXN], dep[MAXN], par[MAXN][MAXD];
vector<wv> adj[MAXN];

void input() {
    cin >> n >> m;
    for (int i = 0; i < n - 1; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({ w, v });
        adj[v].push_back({ w, u });
    }
}

void dfs(int v, int prv) {
    par[v][0] = prv;
    dep[v] = dep[prv] + 1;
    for (auto& i : adj[v]) {
        if (i.v == prv) continue;
        dp[i.v] = dp[v] + i.w;
        dfs(i.v, v);
    }
}

int lca(int u, int v) {
    if (dep[u] < dep[v]) swap(u, v);

    int diff = dep[u] - dep[v];
    for (int i = MAXD - 1; i >= 0; i--)
        if (diff & (1 << i)) u = par[u][i];

    if (u == v) return u;

    for (int i = MAXD - 1; i >= 0; i--) {
        if (par[u][i] ^ par[v][i]) {
            u = par[u][i];
            v = par[v][i];
        }
    }
    return par[u][0];
}

void f() {
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        cout << dp[u] + dp[v] - 2 * dp[lca(u, v)] << '\n';
    }
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    input();
    
    dfs(1, 0);

    for (int i = 1; i < MAXD; i++) {
        for (int v = 1; v <= n; v++) {
            par[v][i] = par[par[v][i - 1]][i - 1];
        }
    }

    f();

    return 0;
}