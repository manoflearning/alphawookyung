#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const double pie = 3.14159265;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	cout << fixed; cout.precision(7);

	int T; cin >> T;

	while (T--) {
		int N; cin >> N;
		cout << (double)1 / tan(pie / (2 * N)) << '\n';
	}

	return 0;
}