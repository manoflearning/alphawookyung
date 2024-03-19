#include <iostream>
using namespace std;

int main() {
	int t;
	cin >> t;

	for (int test = 0; test < t; test++) {
		int n;
		cin >> n;

		bool two = false, one = false;

		for (int i = 0; i < n; i++) {
			int x;
			cin >> x;

			if (x % 2 == 0) two = true;
			else one = true;
		}

		if (one) {
			if (two) cout << "YES\n";
			else {
				if (n % 2 == 0) cout << "NO\n";
				else cout << "YES\n";
			}
		}
		else cout << "NO\n";
	}

	return 0;
}