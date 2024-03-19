#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	vector<int> arr(N);
	for (int& i : arr) cin >> i;
	arr.push_back(0);

	sort(arr.begin(), arr.end());

	int flag; cin >> flag;

	int R = lower_bound(arr.begin(), arr.end(), flag) - arr.begin(), L = R - 1;
	
	int ans = 0;
	if (!(flag == arr[L] || flag == arr[R]))
		ans += (flag - arr[L]) * (arr[R] - flag) - 1;

	cout << ans << '\n';

	return 0;
}