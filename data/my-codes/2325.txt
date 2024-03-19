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
		int N, M;
		cin >> N >> M;

		if (N == 1 || M == 1) cout << "YES" << '\n';
		else if (N <= 2 && M <= 2) cout << "YES" << '\n';
		else cout << "NO" << '\n';
	}

	return 0;
}