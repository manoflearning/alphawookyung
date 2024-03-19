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

int k, n;
set<int> dp[9];

void input() {
    cin >> k >> n;
}

void f() {
    int b = 0;
	for (int i = 1; i <= 8; i++) {
	    b = 10 * b + k;
	    dp[i].insert(b);
	}
	
	for (int i = 2; i <= 8; i++) {
	    for (int j = 1; j <= i - 1; j++) {
	        for (auto& c1 : dp[j]) {
	            for (auto& c2 : dp[i - j]) {
	                dp[i].insert(c1 + c2);
	                dp[i].insert(c1 - c2);
	                dp[i].insert(c1 * c2);
	                if (c2 != 0) dp[i].insert(c1 / c2);
	            }
	        }
	    }
	}
}

int query() {
    int a; cin >> a;
    for (int i = 1; i <= 8; i++) {
        if (dp[i].find(a) != dp[i].end()) return i;
    }
    return -1;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	f();
	
	for (int i = 0; i < n; i++) {
	    int ans = query();
	    if (ans == -1) cout << "NO\n";
	    else cout << ans << '\n';
	}
	
	return 0;
}
