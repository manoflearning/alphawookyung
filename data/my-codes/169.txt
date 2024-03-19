#include <bits/stdc++.h>
using namespace std;
#define ll long long

const int MOD = 1e9 + 7;

ll n, m, a, h;

int main() {
    cin >> n >> m >> a >> h;
    
    ll ans = 1;
    for (int i = 0; i < n - 1; i++) {
        ans = (ans * m) % MOD;
    }
    
    cout << ans;
    
    return 0;
}