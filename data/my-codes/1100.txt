#include <iostream>
#include <vector>
#include <cstring>
using namespace std;

const int MAXV = 100000;

vector<int> adj[MAXV + 5];

int par[MAXV + 5];

void dfs(int now, int prv);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	for (int i = 0; i < N - 1; i++) {
		int a, b;
		cin >> a >> b;
		adj[a].push_back(b);
		adj[b].push_back(a);
	}

	dfs(1, 0);

	for (int i = 2; i <= N; i++)
		cout << par[i] << '\n';

	return 0;
}

void dfs(int now, int prv) {
	par[now] = prv;
	for (int next : adj[now]) {
		if (next == prv) continue;
		dfs(next, now);
	}
}
