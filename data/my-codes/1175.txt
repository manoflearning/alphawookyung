#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <cstring>
using namespace std;

const int INF = 1e9 + 7;
const int MAX = 100, MAXV = 2 * MAX;

int N, M, st = 0, en = MAXV + 1;
int c[MAXV + 5][MAXV + 5], f[MAXV + 5][MAXV + 5];
int d[MAXV + 5][MAXV + 5], prv[MAXV + 5];
vector<int> adj[MAXV + 5];
int mFlow, mCost;

void input();
inline int trans(int x);
void mcmf();
void spfa();
void flow();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	mcmf();

	cout << mFlow << '\n' << mCost << '\n';

	return 0;
}

void input() {
	cin >> N >> M;

	for (int i = 1; i <= N; i++) {
		adj[st].push_back(i);
		adj[i].push_back(st);
		cin >> c[st][i];
	}

	for (int i = 1; i <= M; i++) {
		adj[trans(i)].push_back(en);
		adj[en].push_back(trans(i));
		cin >> c[trans(i)][en];
	}

	for (int b = 1; b <= M; b++) {
		for (int a = 1; a <= N; a++) {
			adj[a].push_back(trans(b));
			adj[trans(b)].push_back(a);
			cin >> c[a][trans(b)];
		}
	}

	for (int b = 1; b <= M; b++) {
		for (int a = 1; a <= N; a++) {
			cin >> d[a][trans(b)];
			d[trans(b)][a] = -d[a][trans(b)];
		}
	}
}

inline int trans(int x) {
	return MAX + x;
}

void mcmf() {
	while (true) {
		spfa();

		if (prv[en] == -1) break;

		flow();
	}
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

		for (int next : adj[now]) {
			if (dist[now] + d[now][next] < dist[next] && c[now][next] - f[now][next] > 0) {
				dist[next] = dist[now] + d[now][next];
				prv[next] = now;
				if (!inQ[next]) {
					inQ[next] = true;
					q.push(next);
				}
			}
		}
	}
}

void flow() {
	int block = INF;
	for (int i = en; i != st; i = prv[i]) {
		block = min(block, c[prv[i]][i] - f[prv[i]][i]);
	}
	for (int i = en; i != st; i = prv[i]) {
		mCost += d[prv[i]][i] * block;
		f[prv[i]][i] += block;
		f[i][prv[i]] -= block;
	}
	mFlow += block;
}