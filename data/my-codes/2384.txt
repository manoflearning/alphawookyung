#include <iostream>
#include <algorithm>
using namespace std;

bool cmp(int a, int b) {
	return a > b;
}

int main() {
	cin.tie(NULL);
	cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int t;
	cin >> t;

	while (t--) {
		int a[3];
		cin >> a[0] >> a[1] >> a[2];

		sort(a, a + 3, cmp);

		int ans = 0;
		for (int i = 0; i < 3; i++)
			if (a[i] > 0) {
				ans++;
				a[i]--;
			}
		for (int i = 0; i < 3; i++) {
			for (int l = i + 1; l < 3; l++) {
				if (a[i] > 0 && a[l] > 0) {
					a[i]--, a[l]--;
					ans++;
				}
			}
		}
		if (a[0] > 0 && a[1] > 0 && a[2] > 0) {
			ans++;
		}
		cout << ans << '\n';
	}

	return 0;
}