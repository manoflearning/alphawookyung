#include <iostream>
using namespace std;

int arr[101];

int main() {
	cin.tie(NULL);
	cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int t;
	cin >> t;

	for (int test = 0; test < t; test++) {
		int n, d;
		cin >> n >> d;

		for (int i = 1; i <= n; i++)
			cin >> arr[i];

		int ans = arr[1];

		for (int i = 2; i <= n; i++) {
			if (d < (i - 1)) break;

			if (arr[i] >= d / (i - 1)) {
				ans += d / (i - 1);
				d = 0;
			}
			else {
				ans += arr[i];
				d -= arr[i] * (i - 1);
			}
		}

		cout << ans << '\n';
	}

	return 0;
}