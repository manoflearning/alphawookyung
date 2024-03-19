#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;
	
	while (T--) {
		ll a, K;
		cin >> a >> K;

		for (int i = 0; i < K - 1; i++) {
			int mn = 11, mx = -1;
			for (ll d = 1; d <= a; d *= 10) {
				int n = a % (d * 10) / d;

				mn = min(mn, n);
				mx = max(mx, n);
			}
			
			if (mn == 0) break;

			a += mn * mx;
		}
	
		cout << a << '\n';
	}

	return 0;
}