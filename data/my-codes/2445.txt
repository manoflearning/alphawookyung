#include <bits/stdc++.h>
#define ll long long
using namespace std;

ll lower_bound(ll L, ll R, ll k);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int n, k;
		cin >> n >> k;

		ll flag = lower_bound(2, 1e5, k);

		k -= (flag - 2) * (flag - 1) / 2;

		for (int i = n; i >= 1; i--) {
			if (i == flag || i == k) cout << 'b';
			else cout << 'a';
		}
		cout << '\n';
	}

	return 0;
}

ll lower_bound(ll L, ll R, ll k) {
	if (L == R) return L;

	ll mid = (L + R) / 2;

	if ((mid - 1) * mid / 2 < k) return lower_bound(mid + 1, R, k);
	else lower_bound(L, mid, k);
}