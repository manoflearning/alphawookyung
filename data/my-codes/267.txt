#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
using namespace std;

const int MAXV = 1e5;
const int INF = 1e9 + 7;

struct dv {
	int d, v;
};

struct cmp {
	bool operator()(dv a, dv b) {
		return a.d > b.d;
	}
};

int N, dist[MAXV + 5];
vector<dv> adj[MAXV + 5];

void input();
int dijkstra(int st);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	cout << dist[dijkstra(dijkstra(1))] << '\n';

	return 0;
}

void input() {
	cin >> N;

	for (int i = 0; i < N; i++) {
		int u; cin >> u;
		while (true) {
			int v; cin >> v;
			if (v == -1) break;
			int d; cin >> d;
			adj[u].push_back({ d, v });
		}
	}
}

int dijkstra(int st) {
	fill(dist + 1, dist + 1 + N, INF);

	int ret = 0;

	priority_queue<dv, vector<dv>, cmp> pq;
	pq.push({ 0, st });
	dist[st] = 0;

	while (!pq.empty()) {
		int now = pq.top().v, d = pq.top().d;
		pq.pop();

		if (dist[ret] < dist[now]) ret = now;

		for (auto& e : adj[now]) {
			int next = e.v;
			if (dist[next] > dist[now] + e.d) {
				dist[next] = dist[now] + e.d;
				pq.push({ dist[next], next });
			}
		}
	}

	return ret;
}