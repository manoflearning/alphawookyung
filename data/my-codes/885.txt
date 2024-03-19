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

	sort(arr.begin(), arr.end());

	int psum = 0;

	for (int i = 0; i < N; i++) {
		if (psum + 2 <= arr[i]) break;
		psum += arr[i];
	}

	cout << psum + 1;

	return 0;
}