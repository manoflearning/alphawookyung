#include <iostream>
#include <vector>
#include <cstring>
#include <queue>
#include <algorithm>
using namespace std;

const int MAX = 1000, MAXV = 2 * MAX;
const bool IN = true, OUT = false;
const int INF = 1e9 + 7;

struct edge {
	int v, c, f, d;
	edge* rev;
	edge(int v1, int c1, int d1): v(v1), c(c1), f(0), d(d1), rev(nullptr){}
	int residual() {
		return c - f;
	}
	void addFlow(int f1) {
		f += f1;
		rev->f -= f1;
	}
};

int V, st, en, prv[MAXV + 5];
vector<edge*> adj[MAXV + 5];
edge* path[MAXV + 5];

void init();
void input();
inline int trans(int x, bool isIN);
int mcmf();
void spfa();
int flow();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	while (cin >> V) {
		init();

		input();

		int ans = mcmf();
		cout << ans << '\n';
	}

	return 0;
}

void init() {
	for (int i = 0; i < MAXV + 5; i++)
		adj[i].clear();
}

void input() {
	int E; cin >> E;

	st = trans(1, IN), en = trans(V, IN);

	for (int i = 1; i <= V; i++) {
		edge* e1 = new edge(trans(i, OUT), 1, 0), * e2 = new edge(trans(i, IN), 0, 0);
		e1->rev = e2, e2->rev = e1;
		if (i == 1) e1->c = 2;
		adj[trans(i, IN)].push_back(e1);
		adj[trans(i, OUT)].push_back(e2);
	}

	for (int i = 0; i < E; i++) {
		int u, v, d;
		cin >> u >> v >> d;

		edge* e1 = new edge(trans(v, IN), 1, d), * e2 = new edge(trans(u, OUT), 0, -d);
		e1->rev = e2, e2->rev = e1;
		adj[trans(u, OUT)].push_back(e1);
		adj[trans(v, IN)].push_back(e2);
	}
}

inline int trans(int x, bool isIN) {
	return isIN ? x : MAX + x;
}

int mcmf() {
	int ret = 0;
	while (true) {
		spfa();

		if (prv[en] == -1) break;

		ret += flow();
	}
	return ret;
}

void spfa() {
	memset(prv, -1, sizeof(prv));
	vector<int> dist(MAXV + 5, INF);
	vector<bool> inQ(MAXV + 5);

	queue<int> q;
	q.push(st);
	dist[st] = 0, inQ[st] = true;

	while (!q.empty()) {
		int now = q.front();
		q.pop();

		inQ[now] = false;

		for (auto* e : adj[now]) {
			int next = e->v;
			if (dist[now] + e->d < dist[next] && e->residual() > 0) {
				dist[next] = dist[now] + e->d;
				prv[next] = now;
				path[next] = e;
				if (!inQ[next]) {
					inQ[next] = true;
					q.push(next);
				}
			}
		}
	}
}

int flow() {
	int ret = 0, block = INF;
	for (int i = en; i != st; i = prv[i]) {
		block = min(block, path[i]->residual());
	}
	for (int i = en; i != st; i = prv[i]) {
		ret += path[i]->d * block;
		path[i]->addFlow(block);
	}
	return ret;
}