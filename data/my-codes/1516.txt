#include <iostream>
#include <vector>
#include <cstring>
#include <queue>
#include <algorithm>
using namespace std;

const int INF = 1e9 + 7;
const int MAX = 100, MAXV = 4 * MAX;

int st = 0, en = MAXV + 1;
vector<int> adj[MAXV + 5];
int c[MAXV + 5][MAXV + 5], f[MAXV + 5][MAXV + 5];
int level[MAXV + 5], work[MAXV + 5];

void input();
int dinic();
void bfs();
int dfs(int now, int flow);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	int ans = dinic();
	cout << ans << '\n';

	return 0;
}

void input() {
	int N, F, D;
	cin >> N >> F >> D;

	for (int i = 1; i <= F; i++) {
		adj[st].push_back(i);
		adj[i].push_back(st);
		c[st][i] = 1;
	}

	for (int i = 1; i <= N; i++) {
		int Fc, Dc;
		cin >> Fc >> Dc;

		while (Fc--) {
			int x; cin >> x;
			adj[x].push_back(MAX + i);
			adj[MAX + i].push_back(x);
			c[x][MAX + i] = 1;
		}

		adj[MAX + i].push_back(2 * MAX + i);
		adj[2 * MAX + i].push_back(MAX + i);
		c[MAX + i][2 * MAX + i] = 1;

		while (Dc--) {
			int x; cin >> x;
			adj[2 * MAX + i].push_back(3 * MAX + x);
			adj[3 * MAX + x].push_back(2 * MAX + i);
			c[2 * MAX + i][3 * MAX + x] = 1;
		}
	}

	for (int i = 1; i <= D; i++) {
		adj[3 * MAX + i].push_back(en);
		adj[en].push_back(3 * MAX + i);
		c[3 * MAX + i][en] = 1;
	}
}

int dinic() {
	int ret = 0;
	while (true) {
		bfs();

		if (level[en] == -1) break;

		memset(work, 0, sizeof(work));

		while (true) {
			int flow = dfs(st, INF);
			if (flow == 0) break;
			ret += flow;
		}
	}
	return ret;
}

void bfs() {
	memset(level, -1, sizeof(level));

	queue<int> q;
	q.push(st);
	level[st] = 0;

	while (!q.empty()) {
		int now = q.front();
		q.pop();

		for (int next : adj[now]) {
			if (level[next] == -1 && c[now][next] - f[now][next] > 0) {
				level[next] = level[now] + 1;
				q.push(next);
			}
		}
	}
}

int dfs(int now, int flow) {
	if (now == en) return flow;

	for (int& i = work[now]; i < adj[now].size(); i++) {
		int next = adj[now][i];
		if (level[next] == level[now] + 1 && c[now][next] - f[now][next] > 0) {
			int df = dfs(next, min(flow, c[now][next] - f[now][next]));
			if (df > 0) {
				f[now][next] += df;
				f[next][now] -= df;
				return df;
			}
		}
	}
	return 0;
}