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

int n;
ll dp[31][61];

void init() {
    for (int i = 0; i < 31; i++) {
        for (int j = 0; j < 61; j++) {
            dp[i][j] = -1;
        }
    }
}

ll f(int w, int h) {
    ll& ret = dp[w][h];
    if (ret != -1) return ret;
    if (w == 0 && h == 0) return ret = 1;
    
    ret = 0;
    if (w > 0) ret += f(w - 1, h + 1);
    if (h > 0) ret += f(w, h - 1);
    
    return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	init();
	
	while (1) {
	    cin >> n;
	    
	    if (!n) break;
	    
	    cout << f(n, 0) << '\n';
	}
	
	return 0;
}
