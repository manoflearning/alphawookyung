#include <bits/stdc++.h>
using namespace std;
#define ll long long

const int MOD = 1e9 + 7;

ll fac[202020];

void preprocess() {
    fac[0] = 1;
    for (int i = 1; i < 202020; i++) {
        fac[i] = i * fac[i - 1] % MOD;
    }
}

int n, m;
vector<vector<int>> adj;
vector<int> a, vi;

void init() {
    adj.clear();
    a.clear();
    vi.clear();
}

void input() {
    cin >> n >> m;
    adj.resize(n + 1);
    a.resize(n + 1);
    vi.resize(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    for (int i = 1; i <= n; i++) cin >> a[i];
}

ll cnt = 0;
map<int, ll> mp;
void dfs(int v) {
    vi[v] = 1;
    mp[a[v]]++;
    cnt++;
    for (auto& i : adj[v]) {
        if (!vi[i]) dfs(i);
    }
}

ll power(ll x, ll y) {
    if (y == 0) return 1;
    if (y == 1) return x;
    ll res = power(x, y / 2);
    return res * res % MOD * (y & 1 ? x : 1) % MOD;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    preprocess();
    
	int tc; cin >> tc;
	while (tc--) {
	    init();
	    
	    input();
	   
	    ll ans = fac[n];
	    
	    for (int v = 1; v <= n; v++) {
	        if (vi[v]) continue;
	        
	        cnt = 0;
	        mp.clear();
	        
	        dfs(v);
	        
	        ans = (ans * power(fac[cnt], MOD - 2)) % MOD;
	        for (auto& i : mp) {
	            ans = (ans * fac[i.second]) % MOD;
	        }
	    }
	    
	    cout << ans << '\n';
	}
}
