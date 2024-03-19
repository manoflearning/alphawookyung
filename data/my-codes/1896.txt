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

ll n, m;
ll t[123461], a[123461], h[123461];

void input() {
    cin >> n >> m;
    
    for (int i = 0; i < n; i++) {
        cin >> t[i] >> a[i] >> h[i];
    }
}

int f(ll mhp) {
    int ret = 1;
    
    ll aa = m, chp = mhp;
    
    for (int i = 0; i < n; i++) {
        if (t[i] == 1) {
            ll cnt1 = chp / a[i] + (chp % a[i] ? 1 : 0);
            ll cnt2 = h[i] / aa + (h[i] % aa ? 1 : 0);
            
            if (cnt1 >= cnt2) {
                chp -= (cnt2 - 1) * a[i];
            }
            if (cnt1 < cnt2) {
                ret = 0;
                break;
            }
        }
        if (t[i] == 2) {
            aa += a[i];
            chp = min(chp + h[i], mhp);
        }
    }
    
    return ret;
}

ll bs(ll l, ll r) {
    while (l < r) {
        ll mid = (l + r) / 2;
        
        if (f(mid)) r = mid;
        else l = mid + 1;
    }
    return l;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	cout << bs(1, (ll)1e12 * 123456);
	
	return 0;
}
