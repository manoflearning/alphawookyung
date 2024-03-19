#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

const int MAX = 2 * 1e5;
const int INF = 1e9 + 7;

struct dv {
	int d, v;
};

struct cmp {
	bool operator() (dv a, dv b) {
		return a.d > b.d;
	}
};

int st, en, dist[MAX + 5];
vector<dv> adj[MAX + 5];

void find_min();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> st >> en;

	for (int i = 0; i <= MAX; i++) {
		if (2 * i <= MAX) adj[i].push_back({ 0, 2 * i });
		if (0 <= i - 1) adj[i].push_back({ 1, i - 1 });
		if (i + 1 <= MAX) adj[i].push_back({ 1, i + 1 });
	}

	find_min();

	cout << dist[en];

	return 0;
}

void find_min() {
	fill(dist, dist + MAX + 5, INF);

	priority_queue<dv, vector<dv>, cmp> pq;
	pq.push({ 0, st });
	dist[st] = 0;

	while (!pq.empty()) {
		int now = pq.top().v, d = pq.top().d;
		pq.pop();
		
		for (auto e : adj[now]) {
			int next = e.v;
			if (dist[next] > d + e.d) {
				dist[next] = d + e.d;
				pq.push({ dist[next], next});
			}
		}
	}
}