#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

ll fac[(int)1e6 + 5], pow2[(int)1e6 + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int n; cin >> n;

	fac[1] = 1;
	for (int i = 2; i <= n; i++)
		fac[i] = fac[i - 1] * i % MOD;

	pow2[0] = 1;
	for (int i = 1; i <= n; i++)
		pow2[i] = 2 * pow2[i - 1] % MOD;

	ll res = (fac[n] - pow2[n - 1] + MOD) % MOD;

	cout << res;

	return 0;
}