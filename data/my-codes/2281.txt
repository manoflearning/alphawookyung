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
const int MAX = 1e5;

int psum[MAX + 5], cnt[10 * MAX + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;
	while (tc--) {
		int n;
		string s;
		cin >> n >> s;

		ll res = 0;

		for (int i = 0; i <= 10 * n; i++) cnt[i] = 0;
		cnt[psum[0] = n]++;
		for (int i = 0; i < n; i++) {
			psum[i + 1] = s[i] - '0' + psum[i];
			res += cnt[psum[i + 1] - (i + 1)]++;
		}
		cout << res << '\n';
	}

	return 0;
}