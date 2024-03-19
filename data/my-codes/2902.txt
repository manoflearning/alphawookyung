#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const long double PI = 3.141592653589793238462643383279502884197;
const int MAX = 1e4;

int N, ans[MAX + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;

	for (int x = 1; x <= sqrt(N); x++) {
		for (int y = 1; y <= sqrt(N); y++) {
			for (int z = 1; z <= sqrt(N); z++) {
				int val = x * x + y * y + z * z + x * y + y * z + z * x;
				if (val <= N) ans[val]++;
			}
		}
	}

	for (int i = 1; i <= N; i++)
		cout << ans[i] << '\n';

	return 0;
}
