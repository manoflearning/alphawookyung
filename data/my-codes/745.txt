#include <iostream>
#include <vector>
#include <queue>
#include <cstring>
#include <algorithm>
using namespace std;

const int INF = 1e9 + 7;
const int MAX = 500;

int N, st = 0, en = MAX + 1;
vector<int> adj[MAX + 5];
int c[MAX + 5][MAX + 5], f[MAX + 5][MAX + 5];
int level[MAX + 5], work[MAX + 5];
bool isA[MAX + 5];

void input();
int dinic();
void bfs();
int dfs(int now, int flow);
void bfs2();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();
	
	int mCut = dinic();

	cout << mCut << '\n';
	
	bfs2();
	
	for (int i = 1; i <= N; i++)
		if (isA[i]) cout << i << ' ';
	cout << '\n';
	for (int i = 1; i <= N; i++)
		if (!isA[i]) cout << i << ' ';
	cout << '\n';

	return 0;
}

void input() {
	cin >> N;

	for (int i = 1; i <= N; i++) {
		int x; cin >> x;

		if (x == 1) {
			adj[st].push_back(i);
			adj[i].push_back(st);
			c[st][i] = INF;
		}
		else if (x == 2) {
			adj[i].push_back(en);
			adj[en].push_back(i);
			c[i][en] = INF;
		}
	}

	for (int i = 1; i <= N; i++) {
		for (int l = 1; l <= N; l++) {
			int w; cin >> w;
			
			if (w == 0) continue;

			adj[i].push_back(l);
			c[i][l] = w;
		}
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
	level[st] = 0;

	queue<int> q;
	q.push(st);

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
			int df = dfs(next, min(c[now][next] - f[now][next], flow));
			if (df > 0) {
				f[now][next] += df;
				f[next][now] -= df;
				return df;
			}
		}
	}

	return 0;
}

void bfs2() {
	queue<int> q;
	q.push(st);
	isA[st] = true;

	while (!q.empty()) {
		int now = q.front();
		q.pop();

		for (int next : adj[now]) {
			if (!isA[next] && c[now][next] - f[now][next] > 0) {
				isA[next] = true;
				q.push(next);
			}
		}
	}
}