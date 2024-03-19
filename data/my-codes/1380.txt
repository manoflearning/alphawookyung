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
const int MAX = 2e5;

bool operator<(pll a, ll b) {
	return a.second < b;
}

ll N, M, K;
string s[26];
int Q[MAX + 5];
vector<ll> segl[26], segr[26];
vector<ll> rsegl[26], rsegr[26];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> M >> K;
	for (int i = 0; i < K; i++) cin >> s[i];
	for (int i = 0; i < N; i++) {
		char c; cin >> c;
		Q[i] = c - 'A';
	}

	for (int i = 0; i < K; i++) {
		ll prv = 0;
		for (int j = 0; j < M; j++) {
			if (s[i][j] == 'U') {
				segl[i].push_back(prv);
				segr[i].push_back(j);
				rsegl[i].push_back(M - prv);
				rsegr[i].push_back(M - j);
				prv = j + 1;
			}
		}
		sort(rsegl[i].begin(), rsegl[i].end());
		sort(rsegr[i].begin(), rsegr[i].end());
	}

	ll l = M - 1, r = M - 1;
	for (l = M - 1; l > 0; l--) {
		if (s[Q[N - 1]][l - 1] == 'U') break;
	}

	ll res = r - l + 1;
	for (int i = N - 2; i >= 0; i--) {
		if (segr[Q[i]].empty()) break;
		if (l > r) break;

		ll nl = lower_bound(segr[Q[i]].begin(), segr[Q[i]].end(), l) - segr[Q[i]].begin();
		ll nr = lower_bound(rsegr[Q[i]].begin(), rsegr[Q[i]].end(), M - r) - rsegr[Q[i]].begin();

		if (nl == segr[Q[i]].size()) break;
		if (r < segr[Q[i]][nl]) break;

		l = segl[Q[i]][nl], r = M - rsegr[Q[i]][nr];
		res += max(0ll, r - l + 1);
	}

	cout << res;

	return 0;
}