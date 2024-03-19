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
const int MAX = 2e3;

ll n, l;
ll h[MAX+5], w[MAX+5], dp[MAX+5];

ll f(int x){
    ll& ret=dp[x];
    if(ret!=-1) return ret;
    if(x==0) return ret=0;
    
    ll mxh=h[x], sumw=w[x];
    ret=f(x-1)+mxh;
    
    for(int i=x-1; i>=1; i--){
        mxh=max(mxh, h[i]);
        sumw+=w[i];
        if(sumw>l) break;
        
        ret=min(ret, f(i-1)+mxh);
    }
    
    return ret;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    memset(dp, -1, sizeof(dp));
    
    cin>>n>>l;
    for(int i=1; i<=n; i++){
        cin>>h[i]>>w[i];
    }
    
    cout<<f(n);
    
    return 0;    
}
