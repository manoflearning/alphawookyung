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

const int MAX = 150005;

int sp[MAX][18];
ll l[MAX], sum[MAX];

int query(int x, ll toGo) {
	int ret = x;

	for (int i = 17; i >= 0; i--) {
		if (ret > 0 && toGo >= sum[ret] - sum[sp[ret][i]]) {
			int nxt = sp[ret][i];
			toGo -= sum[ret] - sum[nxt];
			ret = nxt;
		}
	}
	return ret;
}

void modify(int m, int x) {
	sum[m] = sum[x] + l[m];
	
	sp[m][0] = x;
	for (int i = 1; i < 18; i++) {
		sp[m][i] = sp[sp[m][i - 1]][i - 1];
	}
}

void solve() {
	int q; cin >> q;

	int m = 1, lAns = 0;
	for (int i = 0; i < q; i++) {
		string s; int a; ll b;
		cin >> s >> a >> b;

		if (s == "query") {
			lAns = query(a, b);
			cout << lAns << '\n';
		}
		if (s == "ad-hoc") {
			int x = (a + lAns) % m;
			l[m] = b;
			modify(m++, x);
		}
	}
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	solve();

	return 0;
}
