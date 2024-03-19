#include <iostream>
#include <vector>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		string p; cin >> p;

		int n; cin >> n;

		vector<int> arr(n);
		char c; cin >> c;
		for (int i = 0; i < n; i++) {
			cin >> arr[i];
			if (i != n - 1) cin >> c;
		}
		cin >> c;

		int L = 0, R = n;
		bool rev = false;
		for (auto& q : p) {
			if (q == 'R') rev = !rev;
			else {
				if (rev) R--;
				else L++;
			}
		}

		if (L <= R) {
			if (rev) {
				cout << '[';
				for (int i = R - 1; i >= L; i--) {
					cout << arr[i];
					if (i != L) cout << ',';
				}
				cout << "]\n";
			}
			else {
				cout << '[';
				for (int i = L; i < R; i++) {
					cout << arr[i];
					if (i != R - 1) cout << ',';
				}
				cout << "]\n";
			}
		}
		else cout << "error\n";
	}

	return 0;
}