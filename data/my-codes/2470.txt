#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;
const int MAX = 2 * 1e5;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int T; cin >> T;

	while (T--) {
		int N, K;
		cin >> N >> K;

		vector<int> arr(N + 5);
		for (int i = 1; i <= N; i++) cin >> arr[i];

		vector<bool> p(N + 5);
		vector<int> psum(N + 5);
		for (int i = 2; i <= N - 1; i++) {
			if (arr[i - 1] < arr[i] && arr[i] > arr[i + 1]) p[i] = true;
			psum[i] = (p[i] ? 1 : 0) + psum[i - 1];
		}

		int mx = -1, flag = -1;
		for (int i = 1; i <= N - K + 1; i++) {
			if (mx < psum[i + K - 2] - psum[i]) {
				mx = psum[i + K - 2] - psum[i];
				flag = i;
			}
		}

		cout << mx + 1 << ' ' << flag << '\n';
	}

	return 0;
}