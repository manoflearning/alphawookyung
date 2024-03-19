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
		int x, n, m;
		cin >> x >> n >> m;

		while (n-- && x >= x / 2 + 10) x = x / 2 + 10;

		x -= m * 10;
		
		if (x <= 0) cout << "YES" << endl;
		else cout << "NO" << endl;
	}

	return 0;
}
