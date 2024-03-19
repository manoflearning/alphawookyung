#include <iostream>
#include <algorithm>
using namespace std;

int arr[(int)1e3];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int n, m; cin >> n >> m;
		int total = 0;

		for (int i = 0; i < n; i++) {
			cin >> arr[i];
			total += arr[i];
		}
		cout << min(total, m) << '\n';
	}

	return 0;
}