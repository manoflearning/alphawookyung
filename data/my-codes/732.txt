#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N, K;
	cin >> N >> K;
	
	vector<int> arr(N), sub;
	for (int& i : arr) cin >> i;
	
	for (int i = 1; i < arr.size(); i++) {
		sub.push_back(arr[i] - arr[i - 1]);
	}

	sort(sub.begin(), sub.end());

	int ans = arr.back() - arr.front();
	while (--K) {
		ans -= sub.back();
		sub.pop_back();
	}

	cout << ans;

	return 0;
}