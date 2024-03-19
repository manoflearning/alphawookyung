#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		if ((N / 2) % 2 == 1) {
			cout << "NO\n";
			continue;
		}

		cout << "YES\n";
		for (int i = 1; i <= N / 2; i++)
			cout << 2 * i << ' ';
		for (int i = 1; i <= N / 2 / 2; i++)
			cout << 2 * i - 1 << ' ';
		for (int i = N / 2 / 2 + 1; i <= N / 2; i++)
			cout << 2 * i + 1 << ' ';
		cout << '\n';
	}

	return 0;
}