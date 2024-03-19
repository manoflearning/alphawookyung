#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		vector<int> arr(N);
		for (int& i : arr) cin >> i;

		sort(arr.begin(), arr.end());
		
		int odd = 0, even = 0, diff1 = 0;
		for (int i = 0; i < N; i++) {
			if (arr[i] % 2 == 1) odd++;
			else even++;

			if (i != N - 1 && arr[i] + 1 == arr[i + 1]) diff1++;
		}

		if (diff1 > 0) {
			if (odd % 2 == 0 && even % 2 == 0) cout << "YES\n";
			else if (odd % 2 == 1 && even % 2 == 1) cout << "YES\n";
			else cout << "NO\n";
		}
		else {
			if (odd % 2 == 0 && even % 2 == 0) cout << "YES\n";
			else cout << "NO\n";
		}
	}

	return 0;
}