#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int n, x[1005], y[1005], psum[2005][2005];
vi a;

void input() {
    cin >> n;
    for (int i = 0; i < n; i++) {
        cin >> x[i] >> y[i];
        a.push_back(x[i]);
        a.push_back(y[i]);
    }
}

void comp() {
    sort(a.begin(), a.end());
    a.erase(unique(a.begin(), a.end()), a.end());
}

void makingPsum() {
    for (int i = 0; i < n; i++) {
        int yy = lower_bound(a.begin(), a.end(), y[i]) - a.begin();
        int xx = lower_bound(a.begin(), a.end(), x[i]) - a.begin();
        psum[yy][xx] = 1;
    }
    for (int i = 1; i <= 2000; i++) {
        psum[i][0] += psum[i - 1][0];
        psum[0][i] += psum[0][i - 1];
    }
    for (int i = 1; i <= 2000; i++) {
        for (int j = 1; j <= 2000; j++) {
            psum[i][j] += psum[i - 1][j] + psum[i][j - 1] - psum[i - 1][j - 1];
        }
    }
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();

	comp();
	
	makingPsum();
	
	int ans = INF;
	
	for (int i = 1; i < 2000; i++) {
	    for (int j = 1; j < 2000; j++) {
	        int res = 0;
	        res = max(res, psum[i - 1][j - 1]);
	        res = max(res, psum[2000][j - 1] - psum[i - 1][j - 1]);
	        res = max(res, psum[i - 1][2000] - psum[i - 1][j - 1]);
	        res = max(res, psum[2000][2000] - psum[2000][j - 1] - psum[i - 1][2000] + psum[i - 1][j - 1]);
	        ans = min(ans, res);
	    }
	}
	
	cout << ans;
	
	return 0;
}
