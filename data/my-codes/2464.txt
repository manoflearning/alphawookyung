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
const int MAXn = 1e5;

struct vidx {
	int v, idx;
};

int n;
ll S;
vector<vidx> adj[MAXn + 5];
ll w[MAXn + 5], cnt[MAXn + 5];
int cost[MAXn + 5];
ll nowS;
int res;

ll cal(int i, int op) {
	if (op == 1) return cnt[i] * (w[i] - w[i] / 2);
	if (op == 2) return cnt[i] * (w[i] / 2 - w[i] / 4);
}

struct cmp {
	bool operator()(int a, int b) {
		return cal(a, 1) < cal(b, 1);
	}
};

priority_queue<int, vector<int>, cmp> pq1, pq2;

void init() {
	for (int i = 0; i <= n; i++) {
		adj[i].clear();
		w[i] = cnt[i] = cost[i] = 0;
	}
	n = S = nowS = res = 0;
	while (!pq1.empty()) pq1.pop();
	while (!pq2.empty()) pq2.pop();
}

ll dfs(int now, int prv, int idx) {
	ll& ret = cnt[idx];
	for (auto& i : adj[now]) {
		if (i.v == prv) continue;
		ret += dfs(i.v, now, i.idx);
	}
	if (ret == 0) ret = 1;
	nowS += ret * w[idx];
	return ret;
}

void f(priority_queue<int, vector<int>, cmp>& pq) {
	int val = pq.top();
	pq.pop();

	res += cost[val];

	nowS -= cal(val, 1);
	w[val] /= 2;

	pq.push(val);
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int tc; cin >> tc;
	while (tc--) {
		cin >> n >> S;

		for (int i = 0; i < n - 1; i++) {
			int u, v;
			cin >> u >> v >> w[i] >> cost[i];
			adj[u].push_back({ v, i });
			adj[v].push_back({ u, i });
		}

		dfs(1, 0, n);

		for (int i = 0; i < n - 1; i++) {
			if (cost[i] == 1) pq1.push(i);
			else pq2.push(i);
		}
		
		while (nowS > S) {
			if (pq1.empty() || pq2.empty()) {
				if (pq1.empty()) f(pq2);
				else f(pq1);
			}
			else if (nowS - cal(pq1.top(), 1) <= S) f(pq1);
			else {
				ll sum1 = cal(pq1.top(), 1), sum2 = cal(pq2.top(), 1);
				
				int tmp = pq1.top();
				pq1.pop();
				sum1 += max(cal(tmp, 2), (pq1.empty() ? 0 : cal(pq1.top(), 1)));
				pq1.push(tmp);

				if (sum1 > sum2) f(pq1);
				else f(pq2);
			}
		}

		cout << res << '\n';

		init();
	}

	return 0;
}