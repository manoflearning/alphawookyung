#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

const int MAX = 1e4;

int N;
vector<int> adj[MAX + 5];
int w[MAX + 5], dist[MAX + 5], indeg[MAX + 5];

int bfs();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;

	for (int v = 1; v <= N; v++) {
		cin >> w[v];

		int cnt; cin >> cnt;
		indeg[v] = cnt;
		while (cnt--) {
			int x; cin >> x;
			adj[x].push_back(v);
		}
	}

	cout << bfs();

	return 0;
}

int bfs() {
	queue<int> q;

	for (int v = 1; v <= N; v++) {
		if (indeg[v] == 0) q.push(v);
		dist[v] = w[v];
	}

	int ret = 0;
	while (!q.empty()) {
		int now = q.front();
		q.pop();

		ret = max(ret, dist[now]);

		for (int next : adj[now]) {
			dist[next] = max(dist[now] + w[next], dist[next]);
			indeg[next]--;
			if (indeg[next] == 0) {
				q.push(next);
			}
		}
	}
	return ret;
}