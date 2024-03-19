#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int D, N;
	cin >> D >> N;
	
	vector<int> arr(D);
	for (int& i : arr) cin >> i;

	for (int i = 1; i < D; i++) {
		arr[i] = min(arr[i - 1], arr[i]);
	}

	int L = D;
	for (int i = 0; i < N; i++) {
		int x; cin >> x;

		int pos = lower_bound(arr.rbegin(), arr.rend(), x) - arr.rbegin();
		pos = (D - 1) - pos;
		pos = min(pos, L - 1);

		if (pos < 0) {
			L = -1;
			break;
		}

		L = pos;
	}

	cout << L + 1 << '\n';

	return 0;
}