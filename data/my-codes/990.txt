#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	vector<int> arr(N);
	for (int& i : arr) cin >> i;

	int diff = 2 * 1e9 + 7, L = 0, R = 0;
	for (int l = 0; l < N; l++) {
		int r2 = lower_bound(arr.begin(), arr.end(), -arr[l]) - arr.begin(), r1 = r2 - 1;
		if (r1 == l) r1--;
		if (r2 == l) r2++;

		if (0 <= r1 && fabs(arr[l] + arr[r1]) < diff) {
			diff = fabs(arr[l] + arr[r1]);
			L = l, R = r1;
		}
		if (r2 != N && fabs(arr[l] + arr[r2]) < diff) {
			diff = fabs(arr[l] + arr[r2]);
			L = l, R = r2;
		}
	}

	if (L > R) swap(L, R);
	cout << arr[L] << ' ' << arr[R] << '\n';

	return 0;
}