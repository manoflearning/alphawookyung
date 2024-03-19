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

int n, st, en;
vector<int> adj[105];
int dist[105];

int bfs() {
    fill(dist, dist + 105, INF);

    queue<int> q;
    q.push(st);
    dist[st]=0;

    while(!q.empty()) {
        int v=q.front();
        q.pop();

        if(v==en) return dist[v];

        for(auto i : adj[v]) {
            if(dist[i] > dist[v] + 1) {
                dist[i] = dist[v] + 1;
                q.push(i);
            }
        }
    }
    return -1;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    cin>>n>>st>>en;
    
    int m; cin>>m;
    for(int i=0; i<m; i++) {
        int u, v;
        cin>>u>>v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    cout<<bfs();

    return 0;
}