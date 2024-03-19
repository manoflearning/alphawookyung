#include <iostream>
#include <cmath>
using namespace std;
#define ll long long

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
		
	while (true) {
		int x; cin >> x;

		if (x == 0) break;

		int ans = 0;
		
		if (pow((int)sqrt(x), 2) == x) ans++;
		
		for (int i = 1; i < sqrt(x); i++) {
			int left = x - i * i;
			if (left <= 0 || i > sqrt(left)) continue;

			if (pow((int)sqrt(left), 2) == left) ans++;
		}

		for (int i = 1; i < sqrt(x); i++) {
			for (int l = i; l < sqrt(x); l++) {
				int left = x - (i * i + l * l);
				if (left <= 0) continue;
				if (left <= 0 || l > sqrt(left)) continue;

				if (pow((int)sqrt(left), 2) == left) ans++;
			}
		}

		for (int i = 1; i < sqrt(x); i++) {
			for (int l = i; l < sqrt(x); l++) {
				for (int j = l; j < sqrt(x); j++) {
					int left = x - (i * i + l * l + j * j);
					if (left <= 0) continue;
					if (left <= 0 || j > sqrt(left)) continue;

					if (pow((int)sqrt(left), 2) == left) {
						ans++;
					}
				}
			}
		}

		cout << ans << '\n';
	}

	return 0;
}