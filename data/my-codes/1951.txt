#include <bits/stdc++.h>
using namespace std;
#define ll long long

struct wv {
    ll w; int v;
    bool operator<(const wv& rhs) const {
        return w > rhs.w;
    }
};

int n, m;
vector<wv> adj[50505];
ll dist[50505];

ll dijkstra() {
    priority_queue<wv> pq;
    pq.push({ 0, 1 });
    dist[1] = 0;

    while (pq.size()) {
        int v = pq.top().v;
        ll w = pq.top().w;
        pq.pop();

        if (dist[v] < w) continue;

        for (auto& i : adj[v]) {
            if (dist[i.v] > w + i.w) {
                dist[i.v] = w + i.w;
                pq.push({ w + i.w, i.v });
            }
        }
    }

    return dist[n];
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    for (int i = 0; i < 50505; i++) dist[i] = 1e18;

    cin >> n >> m;
    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({ w, v });
        adj[v].push_back({ w, u });
    }

    cout << dijkstra();

    return 0;
}