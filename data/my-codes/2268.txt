#include <iostream>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int n, m; cin >> n >> m;

		if (n % m == 0) cout << "YES\n";
		else cout << "NO\n";
	}

	return 0;
}