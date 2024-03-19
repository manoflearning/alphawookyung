#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const long double PI = 3.141592653589793238462643383279502884197;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;
	
	while (tc--) {
		ll l, r, m;
		cin >> l >> r >> m;

		ll ansA = -1, ansB = -1, ansC = -1;
		for (ll a = l; a <= r; a++) {
			if (m + (l - r) <= m / a * a) {
				ansA = a;
				ansB = l + m % a;
				ansC = l;
			}
			if (m / a * a + a <= m + (r - l)) {
				ansA = a;
				ansB = l;
				ansC = l + m / a * a + a - m;
			}

			if (ansA != -1) break;
		}

		cout << ansA << ' ' << ansB << ' ' << ansC << '\n';
	}

	return 0;
}