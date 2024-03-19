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

struct ds {
    int d, s, num;
};

bool operator<(ds& a, ds& b) {
    return a.d - a.s > b.d - b.s;
}

int n, c, dp[505][10005];
vector<ds> a;

void init() {
    for (int i = 0; i < 505; i++) {
        for (int j = 0; j < 10005; j++) {
            dp[i][j] = -1;
        }
    }
}

void input() {
    cin >> n >> c;
    for (int i = 0; i < n; i++) {
        int d, s;
        cin >> d >> s;
        a.push_back({ d, s, i + 1 });
    }
}

int f(int idx, int sum) {
    int& ret = dp[idx][sum];
    if (ret != -1) return ret;
    if (idx == n) return ret = 0;
    
    ret = f(idx + 1, sum);
    if (sum + max(a[idx].d, a[idx].s) <= c) {
        ret = max(ret, f(idx + 1, sum + a[idx].s) + 1);
    }
    
    return ret;
}

void f2(int idx, int sum, int cnt) {
    if (idx == n) return;
    
    if (cnt == f(idx + 1, sum)) {
        f2(idx + 1, sum, cnt);
    }
    else {
        cout << a[idx].num << ' ';
        f2(idx + 1, sum + a[idx].s, cnt - 1);
    }
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	init();
	
	input();
	
	sort(a.begin(), a.end());
	
	int ans = f(0, 0);
	
	cout << ans << '\n';
	f2(0, 0, ans);
	
	return 0;
}
