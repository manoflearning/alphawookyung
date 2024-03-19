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

	int t; cin >> t;

	while (t--) {
		int c, d;
		cin >> c >> d;

		if (c > d) swap(c, d);
		
		if (c == 0 && d == 0) cout << 0 << '\n';
		else if (c == d) cout << 1 << '\n';
		else if ((d - c) % 2) cout << -1 << '\n';
		else cout << 2 << '\n';
	}

	return 0;
}