#include <iostream>
#include <algorithm>
using namespace std;

int arr[100000];

int main() {
	cin.tie(NULL);
	cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T;
	cin >> T;

	while (T--) {
		int n, x; 
		cin >> n >> x;

		bool ansIs1 = false;
		int MAX = 0;
		for (int i = 0; i < n; i++) {
			cin >> arr[i];

			if (arr[i] == x) ansIs1 = true;
			MAX = max(MAX, arr[i]);
		}

		if (ansIs1) cout << "1\n";
		else {
			if (MAX > x) cout << "2\n";
			else {
				if (x % MAX == 0) cout << x / MAX << '\n';
				else cout << x / MAX + 1 << '\n';
			}
		}
	}

	return 0;
}