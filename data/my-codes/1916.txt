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

int f, s, g, d[2];
int dist[(int)1e6 + 5];

void init() {
    memset(dist, -1, sizeof(dist));
}

void input() {
    cin >> f >> s >> g >> d[0] >> d[1];
}

int bfs() {
    int ret = 0;
    
    queue<int> q;

    q.push(s);
    dist[s] = 0;
    
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        
        if (v == g) return dist[v];
        
        for (int i = 0; i < 2; i++) {
            int u = v + (i ? -1 : 1) * d[i];
            if (!(1 <= u && u <= f)) continue;
            if (dist[u] != -1) continue;
            
            q.push(u);
            dist[u] = dist[v] + 1;
        }
    }
    
    return -1;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	init();
	
	input();
	
	int ans = bfs();
	
	if (ans == -1) cout << "use the stairs";
	else cout << ans;
	
	return 0;
}
