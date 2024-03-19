#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <queue>
using namespace std;

const int INF = 1e9 + 7;
const int MAX = 200, MAXV = 2 * MAX;

struct xyw {
	int x1, y1, x2, y2, w;
};

int N, M, st = 0, en = MAXV + 1;
xyw wei[MAX + 5], hei[MAX + 5];
int c[MAXV + 5][MAXV + 5], f[MAXV + 5][MAXV + 5];
int d[MAXV + 5][MAXV + 5], prv[MAXV + 5];
vector<int> adj[MAXV + 5];
int mFlow, mCost;

void init();
void input();
void graph_modeling();
inline int trans(int x);
bool isCross(int a, int b);
void mcmf();
void spfa();
void flow();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		init();
		
		input();

		graph_modeling();

		mcmf();

		cout << mFlow << ' ' << -mCost << '\n';
	}

	return 0;
}

void init() {
	memset(c, 0, sizeof(c));
	memset(f, 0, sizeof(f));
	memset(d, 0, sizeof(d));
	for (int i = 0; i < MAXV + 5; i++)
		adj[i].clear();
	mFlow = mCost = 0;
}

void input() {
	cin >> N >> M;

	for (int i = 1; i <= N; i++) {
		cin >> wei[i].x1 >> wei[i].y1 >> wei[i].x2 >> wei[i].y2 >> wei[i].w;
		if (wei[i].x1 > wei[i].x2) swap(wei[i].x1, wei[i].x2);
	}
	for (int i = 1; i <= M; i++) {
		cin >> hei[i].x1 >> hei[i].y1 >> hei[i].x2 >> hei[i].y2 >> hei[i].w;
		if (hei[i].y1 > hei[i].y2) swap(hei[i].y1, hei[i].y2);
	}
}

void graph_modeling() {
	for (int i = 1; i <= N; i++) {
		adj[st].push_back(i);
		adj[i].push_back(st);
		c[st][i] = 1;
	}

	for (int i = 1; i <= M; i++) {
		adj[trans(i)].push_back(en);
		adj[en].push_back(trans(i));
		c[trans(i)][en] = 1;
	}
	
	for (int a = 1; a <= N; a++) {
		for (int b = 1; b <= M; b++) {
			if (isCross(a, b)) {
				adj[a].push_back(trans(b));
				adj[trans(b)].push_back(a);
				c[a][trans(b)] = INF;
				d[a][trans(b)] = -wei[a].w * hei[b].w;
				d[trans(b)][a] = wei[a].w * hei[b].w;
			}
		}
	}
}

inline int trans(int x) {
	return MAX + x;
}

bool isCross(int a, int b) {
	if (!(wei[a].x1 <= hei[b].x1 && hei[b].x1 <= wei[a].x2)) return false;
	if (!(hei[b].y1 <= wei[a].y1 && wei[a].y1 <= hei[b].y2)) return false;
	return true;
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