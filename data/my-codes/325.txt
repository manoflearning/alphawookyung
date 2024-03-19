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

	int N, K;
	cin >> N >> K;

	vector<int> a(N);
	for (int& i : a) cin >> i;

	ll res = 0;
	multiset<int> l, r;

	for (int i = 0; i < N; i++) {
		if (K <= i) {
			if (l.count(a[i - K])) l.erase(l.find(a[i - K]));
			else r.erase(r.find(a[i - K]));
		}

		if (l.size() <= r.size()) {
			if (r.empty() || a[i] < *r.begin()) l.insert(a[i]);
			else {
				l.insert(*r.begin());
				r.erase(r.begin());
				r.insert(a[i]);
			}
		}
		else {
			if (*--l.end() < a[i]) r.insert(a[i]);
			else {
				r.insert(*--l.end());
				l.erase(--l.end());
				l.insert(a[i]);
			}
		}
		
		if (K - 1 <= i) res += *--l.end();
	}

	cout << res;

	return 0;
}