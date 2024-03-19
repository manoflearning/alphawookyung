#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		int n; cin >> n;

		vector<int> cnt(n + 1);
		for (int i = 0; i < n; i++) {
			int x; cin >> x;
			cnt[x]++;
		}
		
		ll res = 0;
		for (int s = 2; s <= 2 * n; s++) {
			ll sum = 0;
			for (int i = 1; i < s; i++) {
				if (i > s - i) break;
				if (i <= n && s - i <= n) {
					if (i == s - i) sum += cnt[i] / 2;
					else sum += min(cnt[i], cnt[s - i]);
				}
			}
			
			res = max(res, sum);
		}

		cout << res << '\n';
	}

	return 0;
}