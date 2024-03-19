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
		ll a, b, c, d;
		cin >> a >> b >> c >> d;

		if (a <= b) cout << b << '\n';
		else {
			if (c > d) {
				ll diff = c - d, left = a - b;
				if (left % diff == 0) cout << b + left / diff * c << '\n';
				else cout << b + (left / diff + 1) * c << '\n';
			}
			else cout << -1 << '\n';
		}
	}

	return 0;
}
