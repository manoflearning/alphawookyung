#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int MAX = 2 * 1e5;
const int INF = 1e9 + 7;

ll arr[MAX + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		for (int i = 0; i < N; i++)
			cin >> arr[i];

		bool neg = (arr[0] < 0) ? true : false;

		ll mxn = -INF, mxp = -INF, ans = 0;
		for (int i = 0; i < N; i++) {
			if (neg) {
				if (arr[i] < 0) {
					mxn = max(mxn, arr[i]);
				}
				else {
					ans += mxn;
					mxn = -INF;
					mxp = arr[i];
					neg = false;
				}
			}
			else {
				if (arr[i] > 0) {
					mxp = max(mxp, arr[i]);
				}
				else {
					ans += mxp;
					mxp = -INF;
					mxn = arr[i];
					neg = true;
				}
			}
		}

		if (mxn != -INF) ans += mxn;
		if (mxp != -INF) ans += mxp;

		cout << ans << '\n';
	}

	return 0;
}