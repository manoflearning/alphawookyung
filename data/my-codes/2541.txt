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
		ll N; cin >> N;

		ll sum = 0;
		for (ll i = 2; i <= N / 2 + 1; i++) {
			sum += (i - 1) * ((i * 2 - 1) * (i * 2 - 1) - ((i - 1) * 2 - 1) * ((i - 1) * 2 - 1));
		}

		cout << sum << '\n';
	}

	return 0;
}