#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const int MAX = 1e3;

struct wv {
    int w, v;
};

bool operator<(wv a, wv b) {
    return a.w > b.w;
}

vector<wv> adj[MAX + 5];
vector<int> dist(MAX + 5, INF);
int dp[MAX + 5];

void dijkstra(int st) {
    priority_queue<wv> pq;

    pq.push({ 0, st });
    dist[st] = 0;

    while (!pq.empty()) {
        int v = pq.top().v, w = pq.top().w;
        pq.pop();

        for (auto& i : adj[v]) {
            int u = i.v;
            if (dist[u] > w + i.w) {
                dist[u] = w + i.w;
                pq.push({ dist[u], u });
            }
        }
    }
}

int f(int v) {
    int& ret = dp[v];
    if (ret != -1) return ret;
    if (v == 2) return ret = 1;
    
    ret = 0;
    for (auto& i : adj[v]) {
        if (dist[i.v] < dist[v]) ret += f(i.v);
    }
    return ret;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    memset(dp, -1, sizeof(dp));
    
    int n, m;
    cin >> n >> m;
    
    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({w, v});
        adj[v].push_back({w, u});
    }
    
    dijkstra(2);
    
    cout << f(1);
    
    return 0;
}
