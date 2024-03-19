#include <iostream>
#include <climits>
#include <cmath>
#include <algorithm>
using namespace std;

const int MAX = 100000;
int arr[MAX];

int main() {
	cin.tie(NULL);
	cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int t;
	cin >> t;

	for (int test = 0; test < t; test++) {
		int n;
		cin >> n;

		int mi = INT_MAX, ma = -1, ans = -1;

		for (int i = 0; i < n; i++) {
			cin >> arr[i];
		}
		for (int i = 0; i < n; i++) {
			if (arr[i] == -1) continue;
			if (i != 0 && arr[i - 1] != -1) ans = max(ans, (int)fabs(arr[i] - arr[i - 1]));
			if ((i == 0 || arr[i - 1] != -1) && (i == n - 1 || arr[i + 1] != -1)) continue;
			mi = min(mi, arr[i]);
			ma = max(ma, arr[i]);
		}

		if (mi == INT_MAX && ma == -1) cout << max(0, ans) << " 0\n";
		else if (mi == INT_MAX) cout << max(0, ans) << ' ' << ma << '\n';
		else if (ma == -1) cout << max(0, ans) << ' ' << mi << '\n';
		else {
			int mid = (mi + ma) / 2;
			cout << max(ans, ma - mid) << ' ' << mid << '\n';
		}
	}

	return 0;
}