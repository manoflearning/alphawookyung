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

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

struct wv {
	int w, v;
};

bool operator<(wv a, wv b) {
	return a.w > b.w;
}

int n, m, dist[25];
vector<wv> adj[25];

void init() {
	for (int i = 0; i < 25; i++) {
		dist[i] = INF;
		adj[i].clear();
	}
}

void input() {
	cin >> n >> m;
	for (int i = 0; i < n; i++) {
		int u, v, w;
		cin >> u >> v >> w;
		adj[u].push_back({ w, v });
		adj[v].push_back({ w, u });
	}
}

void f() {
	priority_queue<wv> pq;

	pq.push({ 0, 0 });
	dist[0] = 0;

	while (!pq.empty()) {
		int v = pq.top().v, w = pq.top().w;
		pq.pop();

		for (auto& i : adj[v]) {
			int u = i.v;
			if (dist[u] > w + i.w) {
				dist[u] = w + i.w;
				pq.push({ dist[u], u });
			}
		}
	}
}

void f2() {
	vector<int> ans(1, m - 1);
	for (int v = m - 1; v != 0; ) {
		for (auto& i : adj[v]) {
			if (dist[i.v] + i.w == dist[v]) {
				v = i.v;
				ans.push_back(v);
				break;
			}
		}
	}

	for (int i = ans.size() - 1; i >= 0; i--) cout << ans[i] << ' ';
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	for (int t = 1; t <= tc; t++) {
		init();

		input();

		f();

		cout << "Case #" << t << ": ";
		if (dist[m - 1] == INF) cout << -1 << '\n';
		else {
			f2();
			cout << '\n';
		}
	}

	return 0;
}