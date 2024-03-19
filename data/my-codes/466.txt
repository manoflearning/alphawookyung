#include <iostream>
#include <vector>
using namespace std;

vector<int> adj[105];
bool knowTruth[55], visited[105];

int dfs(int u);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N, M;
	cin >> N >> M;

	int T; cin >> T;
	for (int i = 0; i < T; i++) {
		int x; cin >> x;
		knowTruth[x] = true;
	}

	for (int i = 1; i <= M; i++) {
		int cnt; cin >> cnt;
		while (cnt--) {
			int x; cin >> x;
			adj[50 + i].push_back(x);
			adj[x].push_back(50 + i);
		}
	}
	
	for (int i = 1; i <= N; i++) {
		if (knowTruth[i] && !visited[i]) dfs(i);
	}

	int ans = 0;
	for (int i = 51; i <= 50 + M; i++) {
		if (!visited[i]) ans++;
	}

	cout << ans << '\n';

	return 0;
}

int dfs(int u) {
	int ret = 0;
	if (u <= 50) ret++;
	visited[u] = true;
	for (int v : adj[u]) {
		if (visited[v]) continue;
		ret += dfs(v);
	}
	return ret;
}