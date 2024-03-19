#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>
#define vs vector<string>
#define vvs vector<vector<string>>
#define fr first
#define sc second

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1, 1, 1, -1, -1 };
const int dx[] = { 1, -1, 0, 0, 1, -1, 1, -1 };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		int n; cin >> n;

		vl a(n);
		for (auto& i : a) cin >> i;

		ll l = 0, r = 0, ans = 0;
		for (int i = 1; i < n - 1; i++) {			
			if (a[i] & 1) {
				if (r) r--, a[i]++;
			}

			if (a[i] >= 2) {
				if (l) {
					ans += l;
					r += l;
					l = 0;
				}

				if (a[i] & 1) {
					if (r) r--, a[i]++;
				}
				
				r += a[i] / 2;
				ans += a[i] / 2;
				a[i] %= 2;
			}

			if (a[i] & 1) l++;
		}

		if (!l) cout << ans << '\n';
		else cout << -1 << '\n';
	}

	return 0;
}
