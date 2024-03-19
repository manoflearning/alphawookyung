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

int n, b;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	cin >> n;
	
	for (int i = max(1, n - 54); i < n; i++) {
	    int sum = i;
	    for (int j = 1; j <= n; j *= 10) {
	        sum += i / j % 10;
	    }
	    if (sum == n) {
	        cout << i;
	        b = 1;
	        break;
	    }
	}
	
	if (b == 0) cout << 0;
	
	return 0;
}
