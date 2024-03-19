#include <iostream>
#include <algorithm>
#define ll long long
using namespace std;

int arr[101];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int n; cin >> n;

		for (int i = 1; i <= n; i++)
			cin >> arr[i];

		sort(arr + 1, arr + n + 1);

		bool yes = true;

		for (int i = 1; i <= n - 1; i++) {
			if ((arr[n] - arr[i]) % 2 == 0) continue;
			else yes = false;
		}

		if (yes) cout << "YES\n";
		else cout << "NO\n";
	}

	return 0;
}