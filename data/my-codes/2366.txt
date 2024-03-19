#include <iostream>
#include <vector>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		int N; cin >> N;

		vector<int> A(N), B(N);
		for (int& i : A) cin >> i;
		for (int& i : B) cin >> i;

		int cnt0 = 0, cnt1 = 0;
		for (int& i : B) {
			if (i == 0) cnt0++;
			else cnt1++;
		}

		if (cnt0 > 0 && cnt1 > 0) cout << "Yes\n";
		else {
			bool Yes = true;
			for (int i = 1; i < A.size(); i++) {
				if (A[i - 1] > A[i]) Yes = false;
			}

			if (Yes) cout << "Yes\n";
			else cout << "No\n";
		}
	}

	return 0;
}