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
const int MAX = 123456;

int n, b[MAX + 5];
ll w[MAX + 5];
vector<int> adj[MAX + 5];

void input() {
    cin >> n;
    
    for (int v = 2; v <= n; v++) {
        char t;
        int a, p;
        
        cin >> t >> a >> p;
        
        w[v] = a;
        b[v] = (t == 'W');
        
        adj[p].push_back(v);
    }
}

ll f(int v) {
    ll res = 0;
    for (int n : adj[v]) {
        res += f(n);
    }
    
    if (b[v]) res = max(res - w[v], 0ll);
    else res += w[v];
    
    return res;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    input();
    
    cout << f(1);
    
    return 0;
}
