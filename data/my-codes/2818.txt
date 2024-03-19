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
const int MAX = 2e5;

int n, m;
vector<ll> h, w;
ll pl[MAX + 5], pr[MAX + 5];

ll bs(ll x) {
    int l=0, r=m-1;
    while(l<r) {
        int mid=(l+r)>>1;
        if(x <= w[mid]) r=mid;
        else l=mid+1;
    }
    
    if(l==0) return fabs(w[l] - x);
    else return min<ll>(fabs(w[l] - x), fabs(w[l-1] - x));
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    cin>>n>>m;
    h.resize(n);
    w.resize(m);
    for(ll& i : h) cin>>i;
    for(ll& i : w) cin>>i;

    sort(h.begin(), h.end());
    sort(w.begin(), w.end());

    pl[1] = h[1] - h[0], pl[2] = h[2] - h[1];
    for(int i=3; i<n; i++) {
        pl[i] = pl[i-2] + h[i] - h[i-1];
    }

    pr[n-2] = h[n-1] - h[n-2], pr[n-3] = h[n-2] - h[n-3];
    for(int i=n-4; i>=0; i--) {
        pr[i] = pr[i+2] + h[i+1] - h[i];
    }

    ll res = bs(h[0]) + pr[1];
    for(int i=2; i<n; i+=2) {
        res = min(res, pl[i-1] + pr[i+1] + bs(h[i]));
    }

    cout<<res;

    return 0;
}