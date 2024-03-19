#include <bits/stdc++.h>
#define ll long long
using namespace std;

const int MAX = 2 * 1e5;

int arr[MAX + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int n; cin >> n;

		for (int i = 1; i <= n; i++)
			cin >> arr[i];

		bool one = true, twin = false, sTwin = false;
		int t1, t2;
		for (int i = 2; i <= n; i++) {
			if (arr[i - 1] != arr[i]) one = false;
			else if (!twin) {
				twin = true;
				t1 = i - 1; t2 = i;
			}
		}
		if (arr[1] == arr[n]) sTwin = true;

		if (one) {
			cout << 1 << '\n';
			for (int i = 1; i <= n; i++)
				cout << 1 << ' ';
			cout << '\n';
		}
		else if (n % 2 == 0) {
			cout << 2 << '\n';
			for (int i = 1; i <= n; i++) {
				if (i % 2 == 0) cout << 1 << ' ';
				else cout << 2 << ' ';
			}
			cout << '\n';
		}
		else {
			if (twin) {
				cout << 2 << '\n';
				for (int i = 1; i <= n; i++) {
					if (i < t2) {
						if (i % 2 == 0) cout << 2 << ' ';
						else cout << 1 << ' ';
					}
					else {
						if (i % 2 == 0) cout << 1 << ' ';
						else cout << 2 << ' ';
					}
				}
				cout << '\n';
			}
			else if (sTwin) {
				cout << 2 << '\n';
				cout << 1 << ' ';
				for (int i = 2; i <= n - 1; i++) {
					if (i % 2 == 0) cout << 2 << ' ';
					else cout << 1 << ' ';
				}
				cout << 1 << ' ';
				cout << '\n';
			}
			else {
				cout << 3 << '\n';
				for (int i = 1; i <= n - 1; i++) {
					if (i % 2 == 0) cout << 1 << ' ';
					else cout << 2 << ' ';
				}
				cout << 3 << '\n';
			}
		}
	}

	return 0;
}