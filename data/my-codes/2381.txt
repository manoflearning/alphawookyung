#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
using namespace std;

const int MAX = 3 * 1e5;

ll a[MAX + 5], b[MAX + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		for (int i = 1; i <= N; i++)
			cin >> a[i] >> b[i];

		int cnt = 0;
		for (int i = 1; i <= N; i++) {
			if (b[i] >= a[i % N + 1]) {
				b[i] = a[i % N + 1];
				cnt++;
			}
		}

		ll ans = 0;
		for (int i = 1; i <= N; i++) {
			if (b[i] < a[i % N + 1]) {
				ans += a[i % N + 1] - b[i];
				a[i % N + 1] = b[i];
			}
		}

		ll rans = 1e15;
		for (int i = 1; i <= N; i++) {
			rans = min(rans, a[i]);
		}

		cout << ans + rans << endl;
	}

	return 0;
}