#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

const int MAXV = 2e5, MAXK = 1e6;
const int INF = 1e9 + 7;

struct wv {
	int w, v;
};

int N, K;
vector<wv> adj[MAXV + 5];
int chk[MAXK + 5], sz[MAXV + 5];
bool used[MAXV + 5];
vector<int> init;

int cd(int now);
int getSize(int now, int prv);
int getCent(int now, int prv, int cnt);
int calAns(int now, int prv, int dist, int dep);
void updateChk(int now, int prv, int dist, int dep);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	fill(chk, chk + MAXK + 1, INF);

	cin >> N >> K;
	for (int i = 0; i < N - 1; i++) {
		int u, v, w;
		cin >> u >> v >> w;
		adj[u].push_back({ w, v });
		adj[v].push_back({ w, u });
	}

	int ans = cd(1);
	cout << (ans == INF ? -1 : ans);

	return 0;
}

int cd(int now) {
	for (int i : init) chk[i] = INF;
	init.clear();

	int cnt = getSize(now, -1);
	int cent = getCent(now, -1, cnt);
	int ret = INF;

	used[cent] = true;
	chk[0] = 0;

	for (auto& e : adj[cent]) {
		if (!used[e.v]) {
			ret = min(ret, calAns(e.v, cent, e.w, 1));
			updateChk(e.v, cent, e.w, 1);
		}
	}

	for (auto& e : adj[cent]) {
		if (!used[e.v]) ret = min(ret, cd(e.v));
	}

	return ret;
}

int getSize(int now, int prv) {
	sz[now] = 1;
	for (auto& e : adj[now]) {
		if (used[e.v] || e.v == prv) continue;
		sz[now] += getSize(e.v, now);
	}
	return sz[now];
}

int getCent(int now, int prv, int cnt) {
	for (auto& e : adj[now]) {
		if (used[e.v] || e.v == prv) continue;
		if (sz[e.v] > cnt / 2) return getCent(e.v, now, cnt);
	}
	return now;
}

int calAns(int now, int prv, int dist, int dep) {
	if (K < dist) return INF;

	int ret = INF;
	if (chk[K - dist] != INF) ret = dep + chk[K - dist];
	
	for (auto& e : adj[now]) {
		if (used[e.v] || e.v == prv) continue;
		ret = min(ret, calAns(e.v, now, dist + e.w, dep + 1));
	}
	return ret;
}

void updateChk(int now, int prv, int dist, int dep) {
	if (K < dist) return;

	chk[dist] = min(chk[dist], dep);
	init.push_back(dist);

	for (auto& e : adj[now]) {
		if (used[e.v] || e.v == prv) continue;
		updateChk(e.v, now, dist + e.w, dep + 1);
	}
}