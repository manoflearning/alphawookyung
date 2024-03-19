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

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	for (int tc = 1; ; tc++) {
	    int l, p, v;
	    cin >> l >> p >> v;
	    
	    if (l == 0 && p == 0 && v == 0) break;
	    
	    int a = v / p, b = v % p;
	    
	    int ans = l * a + min(b, l);
	    
	    cout << "Case " << tc << ": " << ans << '\n';
	}
	
	return 0;
}
