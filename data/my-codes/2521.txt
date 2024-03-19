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
		ll N; cin >> N;
		if (N % 2) cout << N / 2 << endl;
		else cout << N / 2 - 1 << endl;
	}

	return 0;
}