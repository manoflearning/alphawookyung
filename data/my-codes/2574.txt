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
		int N; cin >> N;

		vector<int> arr(N);
		for (int& i : arr) cin >> i;

		sort(arr.begin(), arr.end());
		
		int ans = INF;
		for (int i = 0; i < N - 1; i++) {
			ans = min(ans, arr[i + 1] - arr[i]);
		}

		cout << ans << '\n';
	}

	return 0;
}