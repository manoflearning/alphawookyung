#include <iostream>
#include <cmath>
#define ll long long
using namespace std;

int arr[2 * (int)1e5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int n, m; cin >> n >> m;

	for (int i = 0; i < n; i++)
		cin >> arr[i];

	if (n > m) cout << 0;
	else {
		ll total = 1;

		for (int i = 0; i < n; i++) {
			for (int l = i + 1; l < n; l++) {
				total *= fabs(arr[i] - arr[l]);
				total %= m;
			}
		}

		cout << total;
	}

	return 0;
}