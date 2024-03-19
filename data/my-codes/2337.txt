#include <iostream>
#include <algorithm>
#define ll long long
using namespace std;

const int MAX = 5000;

int arr[MAX + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int n; cin >> n;

		for (int i = 1; i <= n; i++)
			cin >> arr[i];

		bool yes = false;

		for (int i = 1; i <= n; i++) {
			for (int l = i + 2; l <= n; l++) {
				if (arr[i] == arr[l]) {
					yes = true;
					i = l = n + 1;
					break;
				}
			}
		}

		if (yes) cout << "YES\n";
		else cout << "NO\n";
	}

	return 0;
}