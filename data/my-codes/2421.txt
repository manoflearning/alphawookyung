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

		if (n <= 30) cout << "NO\n";
		else {
			cout << "YES\n";
			if (n == 36 || n == 40 || n == 44)
				cout << "6 10 15 " << n - 31 << '\n';
			else cout << "6 10 14 " << n - 30 << '\n';
		}
	}

	return 0;
}