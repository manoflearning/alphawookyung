#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int n, a[] = { 6, 2, 5, 5, 4, 5, 6, 3, 7, 6 };
int mn, mx;
ll dp[55][105];

void init() {
    for (int i = 0; i < 55; i++) {
        for (int j = 0; j < 105; j++) {
            dp[i][j] = -1;
        }
    }
}

void input() {
    cin >> n;
}

ll f(int len, int cnt) {
    ll& ret = dp[len][cnt];
    if (ret != -1) return ret;
    if (len == 0) return ret = !cnt;
    
    ret = 0;
    for (int i = 0; i <= 9; i++) {
        if (cnt < a[i]) continue;
        ret += f(len - 1, cnt - a[i]);
    }
    
    return ret;
}

void f2(int len, int cnt, ll sum) {
    if (len == 0) return;
    
    for (int i = (cnt == n ? 1 : 0); i <= 9; i++) {
        if (cnt >= a[i] && sum <= f(len - 1, cnt - a[i])) {
            cout << i;
            f2(len - 1, cnt - a[i], sum);
            break;
        }
        sum -= f(len - 1, cnt - a[i]);
    }
}

void mnmx() {
    for (int i = 1; ; i++) {
	    if (f(i, n)) {
	        mn = i; break;
	    }
	}
	for (int i = 50; ; i--) {
	    if (f(i, n)) {
	        mx = i; break;
	    }
    }
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	init();
	
	int tc; cin >> tc;
	
	while (tc--) {
	    input();
	    
	    mnmx();
	    
	    f2(mn, n, 1);
	    cout << ' ';
	    f2(mx, n, f(mx, n));
	    cout << '\n';
	}
	
	return 0;
}
