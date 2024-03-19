#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;
const int MAX = 2 * 1e5;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		vector<int> arr(N + 5);
		for (int i = 0; i < N; i++) {
			cin >> arr[i];
		}

		bool Yes = true;

		int mx = N;
		for (int i = 0; i < N; i++) {
			if (!Yes) break;

			if (arr[i] == mx) mx--;
			else if (arr[i] > mx) Yes = false;
			else {
				for (int l = 1; l <= mx - arr[i]; l++) {
					if (arr[i + l] != arr[i] + l) Yes = false;
					if (l == mx - arr[i]) {
						mx = arr[i] - 1;
						i = i + l;
					}
				}
			}
		}

		if (Yes) cout << "Yes\n";
		else cout << "No\n";
	}

	return 0;
}