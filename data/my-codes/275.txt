#include <iostream>
#include <cassert>
using namespace std;
#define ll long long

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		ll x, y;
		cin >> x >> y;

		ll len = y - x;

		ll L = 0, R = (1 << 20);
		while (L < R) {
			ll mid = (L + R + 1) / 2;

			if (mid * (mid + 1) <= len) L = mid;
			else R = mid - 1;
		}
		
		if (L * (L + 1) == len) cout << 2 * L << '\n';
		else if (L * (L + 1) + (L + 1) >= len) cout << 2 * L + 1 << '\n';
		else cout << 2 * L + 2 << '\n';
	}

	return 0;
}