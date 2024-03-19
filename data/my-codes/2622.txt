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
		int n, k, z;
		cin >> n >> k >> z;

		vector<int> a(n), psum(n);
		for (int& i : a) cin >> i;

		psum[0] = a[0];
		for (int i = 1; i < n; i++)
			psum[i] = psum[i - 1] + a[i];

		int res = -1;
		for (int i = 0; i <= min(z, k / 2); i++) {
			priority_queue<int> pq;
			
			for (int j = 0; j <= min(n - 2, k - 2 * i); j++) pq.push(a[j] + a[j + 1]);
			
			res = max(res, psum[k - 2 * i] + (i > 0 ? i * pq.top() : 0));
		}

		cout << res << '\n';
	}

	return 0;
}