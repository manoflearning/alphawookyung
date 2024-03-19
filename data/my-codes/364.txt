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

string a;
int n, dp[55];
vector<string> b;

void init() {
    memset(dp, -1, sizeof(dp));
}

void input() {
    cin >> a >> n;
    b.resize(n);
    for (auto& i : b) cin >> i;
}

int cost(int idx, string s) {
    int ret = 0;
    vector<int> cnt1(26), cnt2(26);
    
    for (int i = 0; i < s.size(); i++) {
        if (a[idx + i] != s[i]) ret++;
        
        cnt1[a[idx + i] - 'a']++, cnt2[s[i] - 'a']++;
    }
    
    for (int i = 0; i < 26; i++) {
        if (cnt1[i] != cnt2[i]) return INF;
    }
    return ret;
}

int f(int idx) {
    int& ret = dp[idx];
    if (ret != -1) return ret;
    if (idx == a.size()) return ret = 0;
    
    ret = INF;
    for (auto& i : b) {
        if (idx + i.size() <= a.size()) {
            ret = min(ret, cost(idx, i) + f(idx + i.size()));
        }
    }
    
    return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	init();
	
	input();
    
    int ans = (f(0) == INF) ? -1 : f(0);
    cout << ans;
	
	return 0;
}
