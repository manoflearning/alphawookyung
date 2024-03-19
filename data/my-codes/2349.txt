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

bool cmp(pll a, pll b) {
	return a.first < b.first;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		int n; cin >> n;

		vector<pll> a(n);
		vector<ll> psum(n + 1);
		for (int i = 0; i < n; i++) cin >> a[i].first;
		for (int i = 0; i < n; i++) cin >> a[i].second;

		sort(a.begin(), a.end(), cmp);

		psum[n - 1] = a[n - 1].second;
		for (int i = n - 2; i >= 0; i--)
			psum[i] = a[i].second + psum[i + 1];

		ll res = psum[0];
		for (int i = 0; i < n; i++) {
			res = min(res, max(a[i].first, psum[i + 1]));
		}

		cout << res << '\n';
	}

	return 0;
}