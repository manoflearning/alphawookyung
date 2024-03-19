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
		ll n, k;
		cin >> n >> k;

		ll l1, r1, l2, r2, l3, r3, l4, r4;
		cin >> l1 >> r1 >> l2 >> r2;

		if (l1 > l2 || (l1 == l2 && r1 > r2)) {
			swap(l1, l2);
			swap(r1, r2);
		}

		l3 = max(l1, l2), r3 = min(r1, r2);
		l4 = min(l1, l2), r4 = max(r1, r2);

		ll sum = max(0ll, n * (r3 - l3));
		ll first = max(0ll, l2 - r1), second = r4 - l4 - max(0ll, r3 - l3);

		ll res = 0;

		for (int i = 0; i < n && sum < k; i++) {
			ll left = k - sum;

			ll val = min(second, left);

			if (i > 0) {
				if (2 * val > first + val) {
					sum += val;
					res += first + val;
				}
				else {
					sum += val;
					res += 2 * val;
				}
			}
			else {
				sum += val;
				res += first + val;
			}
		}

		if (sum < k) res += 2 * (k - sum);

		cout << res << '\n';
	}

	return 0;
}