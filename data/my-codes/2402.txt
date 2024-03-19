#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		vector<int> arr(N);
		for (int& i : arr) cin >> i;

		int prv = 0, mov = 0, L = 0, R = N - 1, A = 0, B = 0;
		while (true) {
			int sum = 0;
			for (; L <= R && sum <= prv; L++)
				sum += arr[L];
			if (sum == 0) break;
			else {
				prv = sum;
				A += sum;
				mov++;
			}

			sum = 0;
			for (; L <= R && sum <= prv; R--)
				sum += arr[R];
			if (sum == 0) break;
			else {
				prv = sum;
				B += sum;
				mov++;
			}
		}

		cout << mov << ' ' << A << ' ' << B << '\n';
	}

	return 0;
}