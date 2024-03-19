#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vl vector<ll>
#define vvi vector<vector<int>>
#define vvl vector<vector<ll>>
using namespace std;

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		ll a, b;
		cin >> a >> b;

		if (a < b) swap(a, b);

		bool canDo = false;
		for (ll i = b; i <= a; i *= 2) {
			if (i == a) {
				canDo = true;
				break;
			}
		}

		if (!canDo) {
			cout << -1 << '\n';
			continue;
		}

		int cnt = 0;
		while (true) {
			if (a == b) break;

			cnt++;
			if (a >= 8 * b) b *= 8;
			else if (a >= 4 * b) b *= 4;
			else b *= 2;
		}

		cout << cnt << '\n';
	}

	return 0;
}