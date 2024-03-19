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

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    ll A, B;
    cin>>A>>B;
    
    ll res = 0;
    for(ll i=2; i/2<=B; i<<=1) {
        res += (B + 1) / i * (i / 2) + max<ll>(0, (B + 1) % i - i / 2);
        res -= A / i * (i / 2) + max<ll>(0, A % i - i / 2);
    }

    cout<<res;

    return 0;
}