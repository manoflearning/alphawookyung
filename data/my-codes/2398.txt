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
		int N, K;
		cin >> N >> K;

		vector<int> ans;
		bool YES = false;

		if (N % 2 == 1 && K % 2 == 1 && N >= K) {
			YES = true;
			for (int i = 0; i < K - 1; i++) ans.push_back(1);
			ans.push_back(N - (K - 1));
		}
		else if (N % 2 == 0 && N >= K) {
			if (K % 2 == 1 && N >= 2 * K) {
				YES = true;
				for (int i = 0; i < K - 1; i++) ans.push_back(2);
				ans.push_back(N - 2 * (K - 1));
			}
			else if ((N - (K - 1)) % 2 == 1) {
				YES = true;
				for (int i = 0; i < K - 1; i++) ans.push_back(1);
				ans.push_back(N - (K - 1));
			}
		}

		if (YES) {
			cout << "YES\n";
			for (int x : ans) cout << x << ' ';
			cout << '\n';
		}
		else cout << "NO\n";
	}

	return 0;
}