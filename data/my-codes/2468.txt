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
		int N, a, b, c, d;
		cin >> N >> a >> b >> c >> d;

		if (N * (a + b) < (c - d)) cout << "No\n";
		else if (N * (a - b) > (c + d)) cout << "No\n";
		else cout << "Yes\n";
	}

	return 0;
}