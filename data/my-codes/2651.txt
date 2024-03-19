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
		ll N, K;
		cin >> N >> K;

		if (N % 2 == 0) cout << N + K * 2 << '\n';
		else {
			int MOD = -1;
			for (int i = 2; i <= N; i++) {
				if (N % i == 0) {
					MOD = i;
					break;
				}
			}

			cout << N + MOD + (K - 1) * 2 << '\n';
		}
	}

	return 0;
}