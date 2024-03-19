#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const long double PI = 3.141592653589793238462643383279502884197;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int n, p, m = 0;
	cin >> n >> p;

	vector<int> a(n), cnt(4005);
	for (int& i : a) {
		cin >> i;
		m = max(m, i);
		cnt[i]++;
	}

	vector<int> ans;

	for (int x = max(m - n + 1, 1); x < m; x++) {
		int v = 0;
		for (int& i : a)
			if (i < x) v++;

		bool YES = true;
		for (int y = x; y < x + n; y++) {
			v += cnt[y];

			if (v % p == 0) {
				YES = false;
				break;
			}

			v--;
		}

		if (YES) ans.push_back(x);
	}

	cout << ans.size() << '\n';
	for (int& i : ans) cout << i << ' ';

	return 0;
}