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

int R, G, B;
ll dp[205][205][205];
ll r[205], g[205], b[205];

ll f(int R, int G, int B) {
	ll& ret = dp[R][G][B];
	if (ret != -1) return ret;
	if (!R + !G + !B >= 2) return ret = 0;

	if (G && B) ret = max(ret, f(R, G - 1, B - 1) + g[G] * b[B]);
	if (R && B) ret = max(ret, f(R - 1, G, B - 1) + r[R] * b[B]);
	if (R && G) ret = max(ret, f(R - 1, G - 1, B) + r[R] * g[G]);

	return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> R >> G >> B;
	for (int i = 1; i <= R; i++) cin >> r[i];
	for (int i = 1; i <= G; i++) cin >> g[i];
	for (int i = 1; i <= B; i++) cin >> b[i];

	sort(r + 1, r + R + 1);
	sort(g + 1, g + G + 1);
	sort(b + 1, b + B + 1);

	for (int i = 0; i < 205; i++) {
		for (int j = 0; j < 205; j++) {
			for (int k = 0; k < 205; k++) {
				dp[i][j][k] = -1;
			}
		}
	}

	cout << f(R, G, B);

	return 0;
}