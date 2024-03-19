#include <iostream>
#include <vector>
#include <queue>
#include <cstring>
using namespace std;

const int INF = 1e9 + 7;
const int MAXV = 1e4;

int N, A[MAXV + 5], B[MAXV + 5], dist[MAXV + 5];
bool used[MAXV + 5];
vector<int> adj[MAXV + 5];

void init();
void input();
int hopcroft();
void bfs(); 
bool dfs(int a);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	while (cin >> N) {
		init();

		input();
		
		int match = hopcroft();
		cout << match << '\n';
	}

	return 0;
}

void init() {
	memset(used, 0, sizeof(used));
	for (auto& i : adj) i.clear();
}

void input() {
	for (int i = 0; i < N; i++) {
		char x;
		int u, cnt;
		cin >> u >> x >> x >> cnt >> x;

		while (cnt--) {
			int v; cin >> v;
			adj[u].push_back(v - N);
		}
	}
}

int hopcroft() {
	int ret = 0;
	memset(A, -1, sizeof(A));
	memset(B, -1, sizeof(B));
	while (true) {
		bfs();

		int flow = 0;
		for (int i = 0; i < N; i++)
			if (!used[i] && dfs(i)) flow++;

		if (flow == 0) break;
		ret += flow;
	}
	return ret;
}

void bfs() {
	queue<int> q;
	for (int i = 0; i < N; i++) {
		if (!used[i]) {
			dist[i] = 0;
			q.push(i);
		}
		else dist[i] = INF;
	}

	while (!q.empty()) {
		int a = q.front();
		q.pop();
		for (int b : adj[a]) {
			if (B[b] != -1 && dist[B[b]] == INF) {
				dist[B[b]] = dist[a] + 1;
				q.push(B[b]);
			}
		}
	}
}

bool dfs(int a) {
	for (int b : adj[a]) {
		if (B[b] == -1 || (dist[B[b]] == dist[a] + 1 && dfs(B[b]))) {
			used[a] = true;
			A[a] = b;
			B[b] = a;
			return true;
		}
	}
	return false;
}