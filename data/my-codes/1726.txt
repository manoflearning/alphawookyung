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

int R, C, a[4][4];
bool isRow[4][4];

int f(int y, int x);
int cal();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> R >> C;
	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			char c; cin >> c;
			a[y][x] = c - '0';
		}
	}

	cout << f(0, 0);

	return 0;
}

int f(int y, int x) {
	if (y == R) return cal();
	if (x == C) return f(y + 1, 0);

	int ret = 0;
	isRow[y][x] = true;
	ret = max(ret, f(y, x + 1));
	isRow[y][x] = false;
	ret = max(ret, f(y, x + 1));
	return ret;
}

int cal() {
	int ret = 0;
	for (int y = 0; y < R; y++) {
		int sum = 0;
		for (int x = 0; x < C; x++) {
			if (isRow[y][x]) sum = 10 * sum + a[y][x];
			else ret += sum, sum = 0;
		}
		ret += sum;
	}
	for (int x = 0; x < C; x++) {
		int sum = 0;
		for (int y = 0; y < R; y++) {
			if (!isRow[y][x]) sum = 10 * sum + a[y][x];
			else ret += sum, sum = 0;
		}
		ret += sum;
	}
	return ret;
}