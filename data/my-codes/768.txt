#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int T; cin >> T;

	vector<int> arr(10);
	while (T--) {
		for (int i = 0; i < 10; i++)
			cin >> arr[i];
		sort(arr.begin(), arr.end());
		cout << arr[7] << '\n';
	}

	return 0;
}