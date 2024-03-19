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

		vector<int> arr(N + 5), psum(N + 5);
		for (int i = 1; i <= N; i++) {
			cin >> arr[i];
			psum[i] = arr[i] + psum[i - 1];
		}

		vector<bool> check(N + 5);
		for (int i = 1; i <= N; i++) {
			for (int l = i + 1; l <= N; l++) {
				int sum = psum[l] - psum[i - 1];
				if (sum <= N) check[sum] = true;
			}
		}

		int cnt = 0;
		for (int i = 1; i <= N; i++)
			if (check[arr[i]]) cnt++;
		
		cout << cnt << '\n';
	}

	return 0;
}