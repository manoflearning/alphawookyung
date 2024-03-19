#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		ll x, y;
		cin >> x >> y;

		if (x < y) swap(x, y);

		ll a, b;
		cin >> a >> b;

		ll ans = 0;

		if (2 * a < b) ans = a * (x - y) + 2 * a * y;
		else ans = a * (x - y) + b * y;
		cout << ans << '\n';
	}

	return 0;
}