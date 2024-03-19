#include <iostream>
#include <vector>
#include <cstring>
#include <queue>
#include <algorithm>
using namespace std;

const int MAX = 200;
const int MAXV = 2 * MAX;
const int INF = 1e9 + 7;

int N, st, en;
int c[MAXV + 5][MAXV + 5], f[MAXV + 5][MAXV + 5], prv[MAXV + 5];
vector<int> adj[MAXV + 5];
bool visited[MAXV + 5];

void input();
inline int trans(int x);
int max_flow();
void bfs();
int flow();
void bfs2();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	max_flow();

	bfs2();

	for (int i = 1; i <= N; i++) {
		if (visited[i] && !visited[trans(i)]) cout << i << ' ';
	}

	return 0;
}

void input() {
	int M;
	cin >> N >> M;

	cin >> st >> en;
	en = trans(en);

	for (int i = 1; i <= N; i++) {
		adj[i].push_back(trans(i));
		adj[trans(i)].push_back(i);
		cin >> c[i][trans(i)];
	}

	for (int i = 0; i < M; i++) {
		int a, b;
		cin >> a >> b;

		adj[trans(a)].push_back(b);
		adj[b].push_back(trans(a));
		c[trans(a)][b] = INF;

		adj[trans(b)].push_back(a);
		adj[a].push_back(trans(b));
		c[trans(b)][a] = INF;
	}
}

inline int trans(int x) {
	return MAX + x;
}

int max_flow() {
	int ret = 0;
	while (true) {
		bfs();

		if (prv[en] == -1) break;

		ret += flow();
	}
	return ret;
}

void bfs() {
	memset(prv, -1, sizeof(prv));

	queue<int> q;
	q.push(st);
	prv[st] = 0;

	while (!q.empty() && prv[en] == -1) {
		int now = q.front();
		q.pop();
		
		for (int next : adj[now]) {
			if (prv[next] == -1 && c[now][next] - f[now][next] > 0) {
				prv[next] = now;
				q.push(next);
			}
		}
	}
}

int flow() {
	int n = en, block = INF;
	while (n != st) {
		block = min(block, c[prv[n]][n] - f[prv[n]][n]);
		n = prv[n];
	}
	n = en;
	while (n != st) {
		f[prv[n]][n] += block;
		f[n][prv[n]] -= block;
		n = prv[n];
	}
	return block;
}

void bfs2() {
	queue<int> q;
	q.push(st);
	visited[st] = true;

	while (!q.empty()) {
		int now = q.front();
		q.pop();

		for (int next : adj[now]) {
			if (!visited[next] && c[now][next] - f[now][next] > 0) {
				visited[next] = true;
				q.push(next);
			}
		}
	}
}