#include <iostream>
#include <algorithm>
using namespace std;

int arr[100];

bool cmp(int a, int b) {
	return a > b;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int n; cin >> n;

		for (int i = 0; i < n; i++)
			cin >> arr[i];

		sort(arr, arr + n, cmp);

		for (int i = 0; i < n; i++)
			cout << arr[i] << ' ';
		cout << '\n';
	}

	return 0;
}