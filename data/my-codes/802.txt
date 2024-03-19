#include <iostream>
#include <vector>
#include <cstring>
#include <queue>
using namespace std;

const int INF = 1e9 + 7;
const int MAX = 1e3;

int N, M, A[MAX + 5], B[MAX + 5], dist[MAX + 5];
vector<int> adj[MAX + 5];
bool used[MAX + 5];

void input();
int hopcroft();
void bfs();
bool dfs(int a);
void bfs2();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	int total = hopcroft();

	cout << total << '\n';

	bfs2();

	return 0;
}

void input() {
	cin >> N >> M;

	for (int u = 1; u <= N; u++) {
		int cnt; cin >> cnt;
		while (cnt--) {
			int v; cin >> v;
			adj[u].push_back(v);
		}
	}
}

int hopcroft() {
	memset(A, -1, sizeof(A));
	memset(B, -1, sizeof(B));

	int ret = 0;
	while (true) {
		bfs();

		int flow = 0;
		for (int i = 1; i <= N; i++)
			if (!used[i] && dfs(i)) flow++;

		if (flow == 0) break;
		ret += flow;
	}
	return ret;
}

void bfs() {
	queue<int> q;
	for (int i = 1; i <= N; i++) {
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

void bfs2() {
	vector<bool> visitedA(N + 5), visitedB(M + 5);

	queue<int> q;
	for (int a = 1; a <= N; a++)
		if (!used[a]) q.push(a);
	
	int aCnt = 0, bCnt = 0;
	while (!q.empty()) {
		int a = q.front();
		q.pop();
		
		visitedA[a] = true;
		aCnt++;

		for (int b : adj[a]) {
			if (B[b] == -1 || visitedA[B[b]]) continue;
			if (visitedB[b]) continue;

			visitedB[b] = true;
			bCnt++;
			q.push(B[b]);
		}
	}

	cout << N - aCnt << ' ';
	for (int a = 1; a <= N; a++)
		if (!visitedA[a]) cout << a << ' ';
	cout << '\n';

	cout << bCnt << ' ';
	for (int b = 1; b <= M; b++)
		if (visitedB[b]) cout << b << ' ';
}