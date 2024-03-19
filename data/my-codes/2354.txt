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

	int N, S;
	cin >> N >> S;

	if (2 * N <= S) {
		cout << "YES\n";
		for (int i = 0; i < N - 1; i++)
			cout << 1 << ' ';
		cout << S - (N - 1) << '\n';
		cout << N << '\n';
	}
	else cout << "NO\n";

	return 0;
}