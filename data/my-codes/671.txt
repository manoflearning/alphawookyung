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

int r;

struct se {
	int s, e, idx;
	bool operator<(se a) {
		if (s / r != a.s / r) return s / r < a.s / r;
		return e < a.e;
	}
};

int n, a[100005], m, cnt[1000005];
vector<se> q;
vi ans;

void input() {
	cin >> n;
	for (int i = 1; i <= n; i++) cin >> a[i];
	r = sqrt(n);
	
	cin >> m;
	q.resize(m);
	ans.resize(m);

	for (int i = 0; i < m; i++) {
		cin >> q[i].s >> q[i].e;
		q[i].idx = i;
	}
}

void f() {
	for (int i = q[0].s; i <= q[0].e; i++) cnt[a[i]]++;

	int res = 0;
	for (int i = 1; i <= 1e6; i++) {
		if (cnt[i]) res++;
	}
	ans[q[0].idx] = res;

	for (int i = 1; i < m; i++) {
		for (int j = q[i - 1].s; j < q[i].s; j++) {
			cnt[a[j]]--;
			if (!cnt[a[j]]) res--;
		}
		for (int j = q[i].s; j < q[i - 1].s; j++) {
			if (!cnt[a[j]]) res++;
			cnt[a[j]]++;
		}
		for (int j = q[i - 1].e + 1; j <= q[i].e; j++) {
			if (!cnt[a[j]]) res++;
			cnt[a[j]]++;
		}
		for (int j = q[i].e + 1; j <= q[i - 1].e; j++) {
			cnt[a[j]]--;
			if (!cnt[a[j]]) res--;
		}

		ans[q[i].idx] = res;
	}
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	sort(q.begin(), q.end());

	f();

	for (auto& i : ans) cout << i << '\n';

	return 0;
}
