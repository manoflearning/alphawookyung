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

const ll INF = (ll)1e18 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

struct xtyr {
    int x, y;
    ll t, r;
};

bool operator<(xtyr& a, xtyr& b) {
    return a.x < b.x;
}

int n, s1, s2;
vector<xtyr> a;
ll dp[505][505];

void init() {
    for (int i = 0; i < 505; i++) {
        for (int j = 0; j < 505; j++) {
            dp[i][j] = INF;
        }
    }
}

void input() {
    cin >> n >> s1 >> s2;
    a.resize(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i].x >> a[i].t >> a[i].y >> a[i].r;
    }
}

void f() {
    dp[0][0] = 0;
	for (auto& i : a) {
	    for (int j = s1; j >= 0; j--) {
	        for (int k = s2; k >= 0; k--) {
	            if (dp[j][k] == INF) continue;
	            
	            if (j < s1) {
	                int nj = j + i.x, nk = k;
	                
	                if (s1 < nj) nk += nj - s1, nj = s1;
	                if (s2 < nk) nk = s2;
	                
	                dp[nj][nk] = min(dp[nj][nk], dp[j][k] + i.t);
	            }
	            
	            int nk = min(s2, k + i.y);
	            dp[j][nk] = min(dp[j][nk], dp[j][k] + i.r);
	        }
	    }
	}
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	init();
	
	input();
	
	sort(a.begin(), a.end());
	
	f();
	
	if (dp[s1][s2] == INF) cout << -1;
	else cout << dp[s1][s2];
	
	return 0;
}
