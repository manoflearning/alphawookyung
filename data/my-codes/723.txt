#include <iostream>
#include <vector>
#include <cmath>
#include <queue>
#include <cstring>
#define pii pair<int, int>
using namespace std;

const int INF = 1e9 + 7;
const int MAX = 100;

int N, A[MAX + 5], B[MAX + 5], dist[MAX + 5];
pii stu[MAX + 5], tut[MAX + 5];
vector<int> adj[MAX + 5];
bool used[MAX + 5];

void input();
int bMatch(int L, int R);
void bInit();
void graph_modeling(int flag);
int hopcroft();
void bfs();
bool dfs(int a);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	int ans = bMatch(0, 1e9);
	cout << ans << '\n';

	return 0;
}

void input() {
	cin >> N;
	for (int i = 1; i <= N; i++) 
		cin >> stu[i].first >> stu[i].second;
	for (int i = 1; i <= N; i++)
		cin >> tut[i].first >> tut[i].second;
}

int bMatch(int L, int R) {
	if (L == R) return L;

	bInit();

	int mid = (L + R) / 2;

	graph_modeling(mid);

	int total = hopcroft();
	
	if (total == N) return bMatch(L, mid);
	else return bMatch(mid + 1, R);
}

void bInit() {
	memset(used, 0, sizeof(used));
	for (auto& i : adj) i.clear();
}

void graph_modeling(int flag) {
	for (int a = 1; a <= N; a++) {
		for (int b = 1; b <= N; b++) {
			if (fabs(stu[a].first - tut[b].first) + fabs(stu[a].second - tut[b].second) <= flag) {
				adj[a].push_back(b);
			}
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