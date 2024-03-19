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

ll N, M;
vector<pll> a;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> M;
	a.resize(M);
	for (auto& i : a) cin >> i.first >> i.second;

	sort(a.begin(), a.end());

	ll l = 1, r = 1e18;
	while (l < r) {
		ll mid = (l + r + 1) >> 1;

		ll cnt = 1, prv = a[0].first;
		for (int i = 0; i < a.size(); i++) {
			ll x = (a[i].second - prv) / mid;
			prv += mid * x;
			cnt += x;

			while (i < a.size() - 1 && a[i + 1].second < prv + mid) i++;
			if (i < a.size() - 1) {
				prv = max(a[i + 1].first, prv + mid);
				cnt++;
			}
		}

		if (cnt >= N) l = mid;
		else r = mid - 1;
	}

	cout << l;

	return 0;
}