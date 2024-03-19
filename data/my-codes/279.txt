#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>
#define vs vector<string>
#define vvs vector<vector<string>>
#define fr first
#define sc second

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1, 1, 1, -1, -1 };
const int dx[] = { 1, -1, 0, 0, 1, -1, 1, -1 };

int n, dp[11][10];

void init() {
	for (int i = 0; i < 11; i++) {
		for (int j = 0; j < 10; j++) {
			dp[i][j] = -1;
		}
	}
}

void input() {
	cin >> n;
	n++;
}

int f(int d, int x) {
	int& ret = dp[d][x];
	if (ret != -1) return ret;
	if (d == 1) return ret = 1;

	ret = 0;
	for (int i = 0; i < x; i++) ret += f(d - 1, i);

	return ret;
}

void f2(int d, int x, int n) {
	if (d == 0) return;

	cout << x;
	for (int i = 0; i < x; i++) {
		if (n <= f(d - 1, i)) {
			f2(d - 1, i, n);
			return;
		}

		n -= f(d - 1, i);
	}
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	init();

	input();

	int b = 1;
	for (int i = 1; i < 11; i++) {
		for (int j = 0; j < 10; j++) {
			if (n <= f(i, j)) {
				f2(i, j, n);
				i = 10, j = 10, b = 0;
				break;
			}
			n -= f(i, j);
		}
	}

	if (b) cout << -1;

	return 0;
}
