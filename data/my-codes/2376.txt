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
		int n, m, k;
		cin >> n >> m >> k;

		int ans = min(m, n / k);
		if (m > n / k) ans -= (m - n / k) % (k - 1) ? (m - n / k) / (k - 1) + 1 : (m - n / k) / (k - 1);

		cout << ans << '\n';
	}

	return 0;
}
