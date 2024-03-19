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

		if (N == 2 || N == 3) cout << -1 << '\n';
		else if (N == 4) {
			cout << "2 4 1 3" << '\n';
		}
		else if (N % 2 == 1) {
			for (int i = 1; i <= N; i += 2) cout << i << ' ';
			cout << N - 3 << ' ';
			cout << N - 1 << ' ';
			for (int i = N - 5; i >= 2; i -= 2) cout << i << ' ';
			cout << '\n';
		}
		else {
			for (int i = 1; i <= N; i += 2) cout << i << ' ';
			cout << N - 4 << ' ';
			cout << N << ' ';
			cout << N - 2 << ' ';
			for (int i = N - 6; i >= 2; i -= 2) cout << i << ' ';
			cout << '\n';
		}
	}

	return 0;
}