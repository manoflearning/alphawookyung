#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

vector<int> arr(3);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int n; cin >> n;

	for (int i = 0; i < n; i++)
		cin >> arr[i];
	
	sort(arr.begin(), arr.begin() + n);

	for (int i = 1; i <= arr[0]; i++) {
		bool check = true;
		for (int l = 0; l < n; l++)
			if (arr[l] % i != 0) {
				check = false;
				break;
			}
	
		if (check) cout << i << '\n';
	}

	return 0;
}