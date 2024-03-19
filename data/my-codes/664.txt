#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
    ll n; int k;
    cin >> n >> k;

    ll ans = 0;
    for (int i = 0; i < k - 1; i++) {
        if ((1ll << (i + 1)) - 1 <= n) ans++;
    }
    for (int i = k; ; i++) {
        ll val = (1ll << i) - (1ll << (i - k));
        if (n < val) break;
        ans += min(n - val + 1, 1ll << (i - k));
    }
    cout << ans;
}