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

		for (int i = 1; i <= 1e5; i *= 10) {
			if (N % (i * 10) / i != 0) ans.push_back(N % (i * 10) / i * i);
		}

		cout << ans.size() << '\n';
		for (int x : ans) cout << x << ' ';
		cout << '\n';
	}

	return 0;
}