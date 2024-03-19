#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		vector<int> ans;

		int m = 1, cnt = 1;
		while (N != m) {
			int left = N - m;
			if (2 * cnt <= left / 2) {
				ans.push_back(cnt);
				cnt *= 2;
			}
			else if (2 * cnt < left) {
				int ncnt = left / 2;
				ans.push_back(ncnt - cnt);
				cnt = ncnt;
			}
			else {
				int diff = left - cnt;
				ans.push_back(diff);
				cnt += diff;
			}

			m += cnt;
		}

		cout << ans.size() << '\n';
		for (int x : ans) cout << x << ' ';
		cout << '\n';
	}

	return 0;
}