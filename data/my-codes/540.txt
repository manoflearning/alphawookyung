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

const int MAXN = 2e4;

int n;
ll a[MAXN + 5], b[MAXN + 5], c[MAXN + 5];

void input() {
    cin >> n;
    for (int i = 0; i < n; i++) {
        cin >> a[i] >> c[i] >> b[i];
    }
}

ll bs(ll l, ll r) {
    while (l < r) {
        ll mid = (l + r) / 2, sum = 0;
        
        for (int i = 0; i < n; i++) {
            if (a[i] <= mid) {
                sum += (min(mid, c[i]) - a[i]) / b[i] + 1;
            }
        }
        
        if (sum & 1) r = mid;
        else l = mid + 1;
    }
    return l;
}

int f(int x) {
    int ret = 0;
    for (int i = 0; i < n; i++) {
        if (!(a[i] <= x)) continue;
        if (!(x <= c[i])) continue;
        if ((x - a[i]) % b[i]) continue;
        ret++;
    }
    return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	ll ans = bs(1, (ll)1e10);
	
	if (ans > INT_MAX) cout << "NOTHING";
	else cout << ans << ' ' << f(ans);
	
	return 0;
}
