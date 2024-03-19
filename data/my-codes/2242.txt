#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

int digit[30], sum[30];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	digit[0] = sum[0] = 1;
	for (int i = 1; i < 30; i++) {
		digit[i] = digit[i - 1] * 2;
		sum[i] = sum[i - 1] + digit[i];
	}
	
	int T; cin >> T;

	while (T--) {
		int N; cin >> N;
		for (int i = 1; i < 30; i++) {
			if (N % sum[i] == 0) {
				cout << N / sum[i] << '\n';
				break;
			}
		}
	}

	return 0;
}