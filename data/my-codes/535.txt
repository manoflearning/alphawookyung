#include <iostream>
#include <vector>
#include <queue>
#include <cstring>
#include <algorithm>
using namespace std;

const int MAX = 50, MAXV = 100;

int N, M, aCnt[MAX + 5], bCnt[MAX + 5], cnt;
vector<int> adj[MAXV + 5];
int st = 0, en = MAXV + 1;
int c[MAXV + 5][MAXV + 5], f[MAXV + 5][MAXV + 5], prv[MAXV + 5];

void input();
void graph_modeling();
inline int trans(int x);
int max_flow(int a, int b);
void bfs(int a, int b);
bool canGo(int a, int b, int now, int next);
int flow();
void dictioning();
void print();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	if (cnt == -1) {
		cout << -1 << '\n';
		return 0;
	}

	graph_modeling();

	int mFlow = max_flow(-1, -1);

	if (mFlow != cnt) {
		cout << -1 << '\n';
		return 0;
	}

	dictioning();

	print();

	return 0;
}

void input() {
	cin >> N >> M;

	int aC = 0, bC = 0;

	for (int i = 1; i <= N; i++) {
		cin >> aCnt[i];
		aC += aCnt[i];
	}
	for (int i = 1; i <= M; i++) {
		cin >> bCnt[i];
		bC += bCnt[i];
	}

	if (aC != bC) cnt = -1;
	else cnt = aC;
}

void graph_modeling() {
	for (int i = 1; i <= N; i++) {
		adj[st].push_back(i);
		adj[i].push_back(st);
		c[st][i] = aCnt[i];
	}

	for (int i = 1; i <= M; i++) {
		adj[trans(i)].push_back(en);
		adj[en].push_back(trans(i));
		c[trans(i)][en] = bCnt[i];
	}

	for (int a = 1; a <= N; a++) {
		for (int b = 1; b <= M; b++) {
			adj[a].push_back(trans(b));
			adj[trans(b)].push_back(a);
			c[a][trans(b)] = 1;
		}
	}
}

inline int trans(int x) {
	return MAX + x;
}

int max_flow(int a, int b) {
	int ret = 0;
	while (true) {
		bfs(a, b);

		if (prv[en] == -1) break;
		
		ret += flow();
	}
	return ret;
}

void bfs(int a, int b) {
	memset(prv, -1, sizeof(prv));

	queue<int> q;
	q.push(st);
	prv[st] = 0;

	while (!q.empty() && prv[en] == -1) {
		int now = q.front();
		q.pop();
		
		for (int next : adj[now]) {
			if (!canGo(a, b, now, next)) continue;
			if (prv[next] == -1 && c[now][next] - f[now][next] > 0) {
				prv[next] = now;
				q.push(next);
			}
		}
	}
}

bool canGo(int a, int b, int now, int next) {
	if (!(0 < now && now <= MAX)) return true;
	if (!(MAX < next && next <= MAXV)) return true;
	 
	if (now < a) return false;
	else if (now == a) {
		if (next - MAX <= b) return false;
		else return true;
	}
	else return true;
}

int flow() {
	for (int i = en; i != st; i = prv[i]) {
		f[prv[i]][i] += 1;
		f[i][prv[i]] -= 1;
	}
	return 1;
}

void dictioning() {
	for (int a = 1; a <= N; a++) {
		for (int b = 1; b <= M; b++) {
			if (c[a][trans(b)] - f[a][trans(b)] == 0) {
				f[st][a]--, f[a][st]++;
				f[a][trans(b)]--, f[trans(b)][a]++;
				f[trans(b)][en]--, f[en][trans(b)]++;

				int mFlow = max_flow(a, b);
				
				if (mFlow == 0) {
					f[st][a]++, f[a][st]--;
					f[a][trans(b)]++, f[trans(b)][a]--;
					f[trans(b)][en]++, f[en][trans(b)]--;
				}
			}
		}
	}
}

void print() {
	for (int a = 1; a <= N; a++) {
		for (int b = 1; b <= M; b++) {
			if (c[a][trans(b)] - f[a][trans(b)] == 0) cout << 1;
			else cout << 0;
		}
		cout << '\n';
	}
}