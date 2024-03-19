#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		string s; cin >> s;

		int ans = INF;

		int L1 = 0, L2 = 0, L3 = 0;
		bool is1 = false, is2 = false, is3 = false;
		for (int i = 0; i < s.size(); i++) {
			if (s[i] == '1') {
				L1 = i;
				is1 = true;
			}
			else if (s[i] == '2') {
				L2 = i;
				is2 = true;
			}
			else if (s[i] == '3') {
				L3 = i;
				is3 = true;
			}

			if (is1 && is2 && is3) ans = min(ans, i - min({ L1, L2, L3 }) + 1);
		}

		if (ans == INF) cout << 0 << '\n';
		else cout << ans << '\n';
	}

	return 0;
}