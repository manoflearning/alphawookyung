#include <bits/stdc++.h>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int a, b;
		cin >> a >> b;

		if (a % b == 0) cout << 0 << '\n';
		else cout << b - (a % b) << '\n';
	}

	return 0;
}