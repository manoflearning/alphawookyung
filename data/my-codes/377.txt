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
ll dp[25][11][11][11][11][11];
vi a(5);

void init() {
    for (int i = 0; i < 25; i++) {
        for (int j = 0; j < 11; j++) {
            for (int k = 0; k < 11; k++) {
                for (int a = 0; a < 11; a++) {
                    for (int b = 0; b < 11; b++) {
                        for (int c = 0; c < 11; c++) {
                            dp[i][j][k][a][b][c] = -1;
                        }
                    }
                }
            }
        }
    }
}

void input() {
    cin >> n;
    for (int i = 0; i < n; i++) cin >> a[i];
}

ll f(int p1, int p2, vi a) {
    ll& ret = dp[5 * p1 + p2][a[0]][a[1]][a[2]][a[3]][a[4]];
    if (ret != -1) return ret;
    if (!a[0] && !a[1] && !a[2] && !a[3] && !a[4]) return ret = 1;
    
    ret = 0;
    
    for (int i = 0; i < n; i++) {
        if (!a[i] || p1 == i || p2 == i) continue;
        a[i]--;
        ret += f(i, p1, a);
        a[i]++;
    }
    
    return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	init();
	
	input();
	
	ll ans = 0;
	for (int i = 0; i < n; i++) {
	    for (int j = 0; j < n; j++) {
	        if (i == j) continue;
	        a[i]--, a[j]--;
	        ans += f(i, j, a);
	        a[i]++, a[j]++;
	    }
	}
	
	cout << ans;
	
	return 0;
}
