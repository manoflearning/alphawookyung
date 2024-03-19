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
		ll N, K;
		cin >> N >> K;

		ll flag = 1;
		for (ll i = 1; i * i <= N; i++) {
			if (N % i == 0) {
				if (i <= K) flag = max(flag, i);
				if (N / i <= K) flag = max(flag, N / i);
			}
		}
		
		cout << N / flag << '\n';
	}

	return 0;
}