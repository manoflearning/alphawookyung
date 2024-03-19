#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;
const int MAX = 30000;

ll dp[MAX + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	dp[1] = 2;
	for (int i = 2; i <= MAX; i++)
		dp[i] = (ll)3 * i - 1 + dp[i - 1];
	
	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		int cnt = 0;
		while (true) {
			int flag = upper_bound(dp + 1, dp + MAX + 1, N) - dp;
			flag--;
			if (dp[flag] == 0) break;
			N -= dp[flag];
			cnt++;
		}
		cout << cnt << '\n';
	}

	return 0;
}