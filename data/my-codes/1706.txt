#include <iostream>
#include <vector>
#include <cstring>
#include <queue>
#include <algorithm>
using namespace std;

const int INF = 1e9 + 7;

struct dv {
	int d, v;
};
struct cmp {
	bool operator()(dv a, dv b) {
		return a.d > b.d;
	}
};

vector<dv> adj[105];

int N, M, item[105];
int dist[105];
bool getItem[105];

int dijkstra(int st);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int R;
	cin >> N >> M >> R;

	for (int i = 1; i <= N; i++) cin >> item[i];
	for (int i = 0; i < R; i++) {
		int u, v, d;
		cin >> u >> v >> d;
		adj[u].push_back({ d, v });
		adj[v].push_back({ d, u });
	}

	int ans = -1;
	for (int i = 1; i <= N; i++)
		ans = max(ans, dijkstra(i));

	cout << ans << '\n';

	return 0;
}

int dijkstra(int st) {
	fill(dist + 1, dist + 1 + N, INF);
	memset(getItem, 0, sizeof(getItem));

	priority_queue<dv, vector<dv>, cmp> pq;
	pq.push({ 0, st });
	dist[st] = 0;

	int ret = 0;
	while (!pq.empty()) {
		int now = pq.top().v, d = pq.top().d;
		pq.pop();

		if (!getItem[now]) {
			ret += item[now];
			getItem[now] = true;
		}

		for (auto& e : adj[now]) {
			int next = e.v;
			if (dist[next] < d + e.d || d + e.d > M) continue;
			dist[next] = d + e.d;
			pq.push({ d + e.d, next });
		}
	}
	return ret;
}