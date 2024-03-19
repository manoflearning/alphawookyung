#include <iostream>
#include <algorithm>
#define ll long long
using namespace std;

int arr1[101], arr2[101];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int n; cin >> n;

		for (int i = 0; i < n; i++)
			cin >> arr1[i];
		for (int i = 0; i < n; i++)
			cin >> arr2[i];

		sort(arr1, arr1 + n);
		sort(arr2, arr2 + n);

		for (int i = 0; i < n; i++)
			cout << arr1[i] << ' ';
		cout << '\n';
		for (int i = 0; i < n; i++)
			cout << arr2[i] << ' ';
		cout << '\n';
	}

	return 0; 
}