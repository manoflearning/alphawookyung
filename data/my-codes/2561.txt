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

int n, m, w, len;
vector<vector<int>> a, dp;
vector<int> b;

int f(int x, int bit) {
	int& ret = dp[x][bit];
	if (ret != -1) return ret;
	if (x == len) return ret = 0;

	ret = INF;

	for (int nbit = 0; nbit < (1 << w); nbit++) {
		int no = 0;
		for (int i = 0; i < w - 1; i++) {
			int sum = 0;
			if (!(bit & (1 << i))) sum++;
			if (!(bit & (1 << (i + 1)))) sum++;
			if (!(nbit & (1 << i))) sum++;
			if (!(nbit & (1 << (i + 1)))) sum++;

			if (sum % 2 == 0) no = 1;
		}

		if (no) continue;

		int cnt = __builtin_popcount(b[x] ^ nbit);
		ret = min(ret, f(x + 1, nbit) + cnt);
	}

	return ret;
}

void getb(int n, int m) {
    w = n, len = m;

    b.resize(m);
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            if(a[j][i]) b[i] |= (1<<j);
        }
    }
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	cin >> n >> m;

	a.resize(n, vector<int>(m));
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < m; x++) {
			char c; cin >> c;
			a[y][x] = c - '0';
		}
	}
		
	if (n >= 4 && m >= 4) {
		cout << -1;
		return 0;
	}
	if (n == 1) {
		cout << 0 << '\n';
		return 0;
	}

    if(n > m) getb(m, n);
    else getb(n, m);

    dp.resize(len + 1, vector<int>(1<<w, -1));

	int res = INF;
	for (int i = 0; i < (1 << w); i++) {
		int cnt = __builtin_popcount(b[0] ^ i);
		res = min(res, f(1, i) + cnt);
	}

	cout << res;

	return 0;
}