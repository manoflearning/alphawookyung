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

struct yx {
    int y, x;
};

int n, k;
int dp[55][55];
vector<yx> b[2505];

void init() {
    for (int y = 0; y < 55; y++) {
        for (int x = 0; x < 55; x++) {
            dp[y][x] = -1;
        }
    }
}

void input() {
    cin >> n >> k;
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
            int w; cin >> w;
            b[w].push_back({y, x});
        }
    }
}

int f(int w, int y, int x) {
    int& ret = dp[y][x];
    if (ret != -1) return ret;
    if (w == k) return ret = 0;
    
    ret = INF;
    for (auto& i : b[w + 1]) {
        int d = fabs(i.y - y) + fabs(i.x - x);
        ret = min(ret, d + f(w + 1, i.y, i.x));
    }
    
    return ret;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    init();
    
    input();
    
    int res = INF;
    for (auto& i : b[1])
        res = min(res, f(1, i.y, i.x));
    
    if (res >= INF) cout << -1;
    else cout << res;
    
    return 0;
}
