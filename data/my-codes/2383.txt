#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		ll N, L, R;
		cin >> N >> L >> R;

		ll cnt = 1;
		vector<ll> ans;
		for (int i = 1; i <= N; i++) {
			if (cnt + 2 * (N - i) < L) {
				cnt += 2 * (N - i);
			}
			else {
				for (int l = i; l <= N; l++) {
					for (int j = l + 1; j <= N; j++) {
						ans.push_back(l);
						ans.push_back(j);

						if (ans.size() > R - cnt) {
							l = j = N + 1;
							break;
						}
					}
				}
				break;
			}
		}
		ans.push_back(1);

		for (int i = L - cnt; i <= R - cnt; i++)
			cout << ans[i] << ' ';
		cout << '\n';
	}

	return 0;
}