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

struct mpr {
    ll m, p, r;
};

bool operator<(mpr& a, mpr& b) {
    return a.r * b.p < b.r * a.p;
}

int n, t;
vector<mpr> a;
ll dp[55][100005];

void init() {
    for (int i = 0; i < 55; i++) {
        for (int j = 0; j < 100005; j++) {
            dp[i][j] = -1;
        }
    }
}

void input() {
    cin >> n >> t;
    a.resize(n);
    for (auto& i : a) cin >> i.m;
    for (auto& i : a) cin >> i.p;
    for (auto& i : a) cin >> i.r;
}

ll f(int idx, int b) {
    ll& ret = dp[idx][b];
    if (ret != -1) return ret;
    if (idx == n) return ret = 0;
    
    ret = max(ret, f(idx + 1, b));
    
    auto& i = a[idx];
    if (i.r <= t - b)
        ret = max(ret, f(idx + 1, b + i.r) + i.m - (b + i.r) * i.p);
        
    return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	init();
	
	input();
	
	sort(a.begin(), a.end());
	    
    cout << f(0, 0);
	
	return 0;
}
