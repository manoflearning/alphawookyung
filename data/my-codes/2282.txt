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
		string s; cin >> s;

		int cnt = 0;
		vector<int> a;
		for (int i = 0; i < s.size(); i++) {
			if (s[i] == '0') {
				if (cnt > 0) a.push_back(cnt);
				cnt = 0;
			}
			else cnt++;
		}
		if (cnt > 0) a.push_back(cnt);
		
		sort(a.rbegin(), a.rend());

		ll res = 0;
		for (int i = 0; i < a.size(); i += 2) {
			res += a[i];
		}

		cout << res << '\n';
	}

	return 0;
}