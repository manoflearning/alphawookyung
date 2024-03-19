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

	int tc; cin >> tc;

	while (tc--) {
		int n; cin >> n;
		vector<int> a(n);
		for (int& i : a) cin >> i;

		sort(a.begin(), a.end());

		bool YES = true;
		for (int i = 1; i < n; i++) YES &= (a[i] - a[i - 1] <= 1);

		if (YES) cout << "YES\n";
		else cout << "NO\n";
	}

	return 0;
}