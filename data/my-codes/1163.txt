#include <iostream>
#include <vector>
#include <cstring>
#include <queue>
#include <algorithm>
using namespace std;

const int MAX = 100;
const int MAXV = 2 * MAX;
const int INF = 1e9 + 7;

int N, M, st = 0, en = MAXV + 1;
int c[MAXV + 5][MAXV + 5], f[MAXV + 5][MAXV + 5], d[MAXV + 5][MAXV + 5], prv[MAXV + 5];
vector<int> adj[MAXV + 5];

void input();
inline int trans(int x);
int mcmf();
void spfa();
int flow();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	int ans = mcmf();
	cout << ans << '\n';

	return 0;
}

void input() {
	cin >> N >> M;

	for (int i = 1; i <= N; i++) {
		adj[trans(i)].push_back(en);
		adj[en].push_back(trans(i));
		cin >> c[trans(i)][en];
	}

	for (int i = 1; i <= M; i++) {
		adj[st].push_back(i);
		adj[i].push_back(st);
		cin >> c[st][i];
	}

	for (int i = 1; i <= M; i++) {
		for (int l = 1; l <= N; l++) {
			adj[i].push_back(trans(l));
			adj[trans(l)].push_back(i);
			c[i][trans(l)] = INF;
			cin >> d[i][trans(l)];
			d[trans(l)][i] = -d[i][trans(l)];
		}
	}
}

inline int trans(int x) {
	return MAX + x;
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
	dist[st] = 0;
	inQ[st] = true;

	while (!q.empty()) {
		int u = q.front();
		q.pop();
		inQ[u] = false;

		for (int v : adj[u]) {
			if (dist[u] + d[u][v] < dist[v] && c[u][v] - f[u][v] > 0) {
				dist[v] = dist[u] + d[u][v];
				prv[v] = u;
				if (!inQ[v]) {
					inQ[v] = true;
					q.push(v);
				}
			}
		}
	}
}

int flow() {
	int ret = 0, block = INF;
	for (int i = en; i != st; i = prv[i]) {
		block = min(block, c[prv[i]][i] - f[prv[i]][i]);
	}
	for (int i = en; i != st; i = prv[i]) {
		ret += block * d[prv[i]][i];
		f[prv[i]][i] += block;
		f[i][prv[i]] -= block;
	}
	return ret;
}