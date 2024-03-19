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

const int r = 317;

struct se {
	int s, e, idx;
	bool operator<(se a) {
		if (s / r != a.s / r) return s / r < a.s / r;
		return e < a.e;
	}
};

int n, k, a[101010], m;
list<int> d[101010];
vector<se> q;
int cnt[101010], bucket[r + 5];
vi ans;

void input() {
	cin >> n >> k;
	for (int i = 1; i <= n; i++) cin >> a[i];

	cin >> m;
	q.resize(m);
	ans.resize(m);

	for (int i = 0; i < m; i++) {
		cin >> q[i].s >> q[i].e;
		q[i].idx = i;
	}
}

void add(int idx) {
	auto& dd = d[a[idx]];
	if (dd.size()) {
		int x = dd.back() - dd.front();
		cnt[x]--;
		bucket[x / r]--;
	}

	if (dd.empty() || idx < dd.front()) dd.push_front(idx);
	else dd.push_back(idx);

	int x = dd.back() - dd.front();
	cnt[x]++;
	bucket[x / r]++;
}

void del(int idx) {
	auto& dd = d[a[idx]];

	int x = dd.back() - dd.front();
	cnt[x]--;
	bucket[x / r]--;

	if (idx == dd.front()) dd.pop_front();
	else dd.pop_back();

	if (dd.size()) {
		int x = dd.back() - dd.front();
		cnt[x]++;
		bucket[x / r]++;
	}
}

int query() {
	for (int i = n / r; i >= 0; i--) {
		if (!bucket[i]) continue;
		for (int j = r - 1; j >= 0; j--) {
			if (cnt[i * r + j]) return i * r + j;
		}
	}
}

void f() {
	int s = q[0].s, e = q[0].e;
	for (int i = s; i <= e; i++) add(i);
	ans[q[0].idx] = query();

	for (int i = 1; i < m; i++) {
		while (q[i].s < s) add(--s);
		while (e < q[i].e) add(++e);
		while (s < q[i].s) del(s++);
		while (q[i].e < e) del(e--);
		ans[q[i].idx] = query();
	}
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	sort(q.begin(), q.end());

	f();

	for (auto& i : ans)
		cout << i << '\n';

	return 0;
}
