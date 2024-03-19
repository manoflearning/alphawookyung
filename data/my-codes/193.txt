#include <iostream>
#include <vector>
#include <queue>
#include <cstring>
using namespace std;

const int MAX = 2 * 1e5;

int N, K, dist[MAX + 5], cnt[MAX + 5];
vector<int> adj[MAX + 5];

void bfs();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	for (int i = 0; i <= MAX; i++) {
		if (2 * i <= MAX) adj[i].push_back(2 * i);
		if (0 <= i - 1) adj[i].push_back(i - 1);
		if (i + 1 <= MAX) adj[i].push_back(i + 1);
	}

	cin >> N >> K;

	bfs();
	cout << dist[K] << '\n' << cnt[K] << '\n';

	return 0;
}

void bfs() {
	memset(dist, -1, sizeof(dist));

	queue<int> q;
	q.push(N);
	dist[N] = 0;
	cnt[N] = 1;

	while (!q.empty()) {
		int now = q.front();
		q.pop();

		for (int next : adj[now]) {
			if (dist[next] == -1) {
				dist[next] = dist[now] + 1;
				cnt[next] += cnt[now];
				q.push(next);
			}
			else if (dist[next] == dist[now] + 1) cnt[next] += cnt[now];
		}
	}
}