#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;

int N;

int find(int flag, vector<int>& dp, vector<int>& arr);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		cin >> N;

		vector<int> arr(N + 5);
		for (int i = 1; i <= N; i++) cin >> arr[i];

		vector<int> dp(N + 5, -1);

		int ans = 0;
		for (int i = 1; i <= N; i++)
			ans = max(ans, find(i, dp, arr));
		
		cout << ans << '\n';
	}

	return 0;
}

int find(int flag, vector<int>& dp, vector<int>& arr) {
	int& ret = dp[flag];

	if (ret != -1) return ret;

	ret = 1;

	for (int i = 2 * flag; i <= N; i += flag) {
		if (arr[flag] < arr[i])
			ret = max(ret, find(i, dp, arr) + 1);
	}

	return ret;
}