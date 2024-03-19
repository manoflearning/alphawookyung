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
const int MAX = 1e5;

struct wv{
    ll w;
    int v;
};

int n, lCnt, rn;
vector<wv> adj[MAX+5], t[MAX+5];
int par[MAX+5];
ll pw[MAX+5];
ll dp[MAX+5], dp2[MAX+5], dp3[MAX+5];
ll res;

void init(){
    fill(dp, dp+MAX+5, -1);
    fill(dp2, dp2+MAX+5, -1);
    fill(dp3, dp3+MAX+5, -1);
}

void input(){
    cin>>n;
    
    lCnt=n/2+1;
    
    for(int i=0; i<n-1; i++){
        int u, v, w;
        cin>>u>>v>>w;
        adj[u].push_back({w, v});
        adj[v].push_back({w, u});
    }
}

void dfs(int v, int prv){
    for(auto& i : adj[v]){
        if(i.v==prv) continue;
        par[i.v]=v, pw[i.v]=i.w;
        t[v].push_back(i);
        dfs(i.v, v);
    }
}

void dfs2(int v);
ll f(int v);
ll f2(int v);
ll f3(int v);

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    init();
    
    input();
    
    for(int v=1; v<=n; v++){
        if(adj[v].size()==1){
            dfs(rn=v, 0);
            break;
        }
    }
    
    dfs2(rn);
    
    cout<<res;
    
    return 0;
}

void dfs2(int v){
    res+=f(v)*(lCnt-f(v))*pw[v]*pw[v];
    
    ll sum=f3(v);
    for(auto& i : t[v]){
        sum+=f2(i.v)+i.w*f(i.v);
    }
    
    res+=pw[v]*f2(v)*(lCnt-f(v));
    for(auto& i : t[v]){
        res+=i.w*(sum-f2(i.v)-i.w*f(i.v))*f(i.v);
    }
    
    for(auto& i : t[v]){
        dfs2(i.v);
    }
}

ll f(int v){
    ll& ret=dp[v];
    if(ret!=-1) return ret;
    if(t[v].empty()) return ret=1;
    
    ret=0;
    for(auto& i : t[v]){
        ret+=f(i.v);
    }
    return ret;
}

ll f2(int v){
    ll& ret=dp2[v];
    if(ret!=-1) return ret;
    
    ret=0;
    for(auto& i : t[v]){
        ret+=f(i.v)*i.w;
        ret+=f2(i.v);
    }
    return ret;
}

ll f3(int v){
    ll& ret=dp3[v];
    if(ret!=-1) return ret;
    if(v==rn) return ret=0;
    
    ret=f3(par[v])+pw[v]*(lCnt-f(v));
    for(auto& i : t[par[v]]){
        if(i.v==v) continue;
        ret+=f2(i.v)+i.w*f(i.v);
    }
    return ret;
}
