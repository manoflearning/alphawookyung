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

int n, q, a[1005], psum[1005];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	cin >> n >> q;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        if (i > 1) psum[i] = psum[i - 1] + fabs(a[i] - a[i - 1]);
    }
    
    for (int i = 0; i < q; i++) {
        int l, r; cin >> l >> r;
        cout << psum[r] - psum[l] << '\n';
    }
    
	return 0;
}
