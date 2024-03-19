#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;

ll N, K;

ll bSearch(ll L, ll R);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		cin >> N >> K;

		ll ans = bSearch(0, 1e10);

		cout << ans << '\n';
	}

	return 0;
}

ll bSearch(ll L, ll R) {
	ll mid = (L + R) / 2;
	if (mid - mid / N == K) {
		if (mid % N != 0) return mid;
		else return bSearch(L, mid);
	}
	else if (mid - mid / N > K) return bSearch(L, mid);
	else if (mid - mid / N < K) return bSearch(mid + 1, R);
}