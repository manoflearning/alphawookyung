#include <iostream>
#include <vector>
#include <queue>
#include <cstring>
using namespace std;

const int MAX = 5 * 1e5;

int N, K, dist[MAX + 5][2];
vector<int> adj[MAX + 5];

void graph_modeling();
void bfs();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	graph_modeling();

	cin >> N >> K;

	bfs();

	int ans = -1;
	for (int i = 0; ; i++) {
		K += i;
		if (K > MAX) break;

		if (dist[K][i % 2] <= i) {
			ans = i;
			break;
		}
	}

	if (ans == -1) cout << -1 << '\n';
	else cout << ans << '\n';

	return 0;
}

void graph_modeling() {
	for (int i = 0; i <= MAX; i++) {
		if (i - 1 >= 0) adj[i].push_back(i - 1);
		if (i + 1 <= MAX) adj[i].push_back(i + 1);
		if (2 * i <= MAX) adj[i].push_back(2 * i);
	}
}

void bfs() {
	memset(dist, -1, sizeof(dist));

	struct ov {
		bool odd;
		int v;
	};

	queue<ov> q;
	q.push({ false, N });
	dist[N][false] = 0;

	while (!q.empty()) {
		int now = q.front().v;
		bool odd = q.front().odd;
		q.pop();

		for (int next : adj[now]) {
			if (dist[next][!odd] == -1) {
				dist[next][!odd] = dist[now][odd] + 1;
				q.push({ !odd, next });
			}
		}
	}
}