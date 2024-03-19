#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;

ll GCD(ll a, ll b);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		ll a, b, Q;
		cin >> a >> b >> Q;
		
		if (a > b) swap(a, b);

		ll lcm = (a * b) / GCD(a, b);
		
		while (Q--) {
			ll L, R;
			cin >> L >> R;

			ll ans = R / lcm * (lcm - b);
			ans += max(R % lcm - b + 1, (ll)0);
			ans -= (L - 1) / lcm * (lcm - b);
			ans -= max((L - 1) % lcm - b + 1, (ll)0);

			cout << ans << ' ';
		}
		cout << '\n';
	}

	return 0;
}

ll GCD(ll a, ll b) {
	if (b == 0) return a;
	else return GCD(b, a % b);
}