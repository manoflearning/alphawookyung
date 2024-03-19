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

int k, a[505];
ll psum[505], dp[505][505];

void init() {
    for (int i = 0; i < 505; i++) {
        for (int j = 0; j < 505; j++) {
            dp[i][j] = -1;
        }
    }
    
    psum[0] = a[0];
    for (int i = 1; i < k; i++)
        psum[i] = psum[i - 1] + a[i];
}

void input() {
    cin >> k;
    for (int i = 0; i < k; i++) cin >> a[i];
}

ll f2(int l, int r) {
    return psum[r] - (l > 0 ? psum[l - 1] : 0);
}

ll f(int l, int r) {
    ll& ret = dp[l][r];
    if (ret != -1) return ret;
    if (l == r) return ret = 0;
    
    ret = INF;
    
    for (int i = l; i < r; i++) {
        ret = min(ret, f2(l, i) + f2(i + 1, r) + f(l, i) + f(i + 1, r));
    }
    
    return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int tc; cin >> tc;
	
	while (tc--) {
	    input();
	    
	    init();
	    
	    cout << f(0, k - 1) << '\n';
	}

	return 0;
}
