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
		int n; cin >> n;

		int b = 0, a = 0;
		for (int i = 1, l = 2; i <= n; i++, l *= 2) {
			if (i < n / 2) b += l;
			else if (i < n) a += l;
			else b += l;
		}
		cout << b - a << '\n';
	}

	return 0;
}