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
const int MAX = 2 * 1e5;

int N, dp[MAX + 5];
int cnt, sum[2], now[MAX + 5][2];
string X;

int f(int val);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	memset(dp, -1, sizeof(dp));

	cin >> N >> X;

	for (auto& c : X) cnt += c - '0';

	if (cnt > 1) sum[0] = (X[0] - '0') % (cnt - 1);
	sum[1] = (X[0] - '0') % (cnt + 1);

	for (int i = 1; i < N; i++) {
		if (cnt > 1) sum[0] = (2 * sum[0] + X[i] - '0') % (cnt - 1);
		sum[1] = (2 * sum[1] + X[i] - '0') % (cnt + 1);
	}

	if (cnt > 1) now[N - 1][0] = 1 % (cnt - 1);
	now[N - 1][1] = 1 % (cnt + 1);

	for (int i = N - 2; i >= 0; i--) {
		if (cnt > 1) now[i][0] = 2 * now[i + 1][0] % (cnt - 1);
		now[i][1] = 2 * now[i + 1][1] % (cnt + 1);
	}

	for (int i = 0; i < N; i++) {
		int pcnt = cnt, val = 0;

		if (X[i] == '0') {
			pcnt++;
			val = sum[1] + now[i][1];
		}
		if (X[i] == '1') {
			pcnt--;
			if (pcnt == 0) {
				cout << 0 << '\n';
				continue;
			}
			val = sum[0] - now[i][0];
		}

		cout << f(((val % pcnt) + pcnt) % pcnt) + 1 << '\n';
	}

	return 0;
}

int f(int val) {
	int& ret = dp[val];
	if (ret != -1) return ret;
	if (val == 0) return ret = 0;

	int pcnt = 0;
	for (int i = 0; (1 << i) <= val; i++)
		if (val & (1 << i)) pcnt++;

	return ret = f(val % pcnt) + 1;
}
