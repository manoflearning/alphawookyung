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
		vector<int> arr(N + 5), cnt;

		for (int i = 0; i < N; i++) {
			cin >> arr[i];
			int chk = i + arr[i] % N;
			if (arr[i] % N < 0) chk += N;
			cnt.push_back(chk % N);
		}

		sort(cnt.begin(), cnt.end());

		bool YES = true;
		for (int i = 1; i < cnt.size(); i++) {
			if (cnt[i - 1] + 1 != cnt[i]) {
				YES = false;
			}
		}

		if (YES) cout << "YES\n";
		else cout << "NO\n";
	}

	return 0;
}