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

		int res = INF;
		for (char i = '0'; i <= '9'; i++) {
			for (char j = '0'; j <= '9'; j++) {
				bool first = true;
				int cnt = 0;
				for (int k = 0; k < s.size(); k++) {
					if (first) {
						if (s[k] == i) first = !first;
						else cnt++;
					}
					else {
						if (s[k] == j) first = !first;
						else cnt++;
					}
				}
				if (!first && i != j) cnt++;

				res = min(res, cnt);
			}
		}

		cout << res << '\n';
	}

	return 0;
}