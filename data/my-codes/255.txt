#include <iostream>
#include <vector>
#include <cstring>
using namespace std;

const int MAX = 100, MAXV = MAX * MAX;

int R, C, arr[MAX + 5][MAX + 5], labelX[MAX + 5][MAX + 5], labelY[MAX + 5][MAX + 5];
vector<int> adj[MAXV + 5];
int A[MAXV + 5], B[MAXV + 5];
bool visited[MAXV + 5];

void input();
void graph_modeling();
int bMatch();
bool dfs(int a);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	graph_modeling();

	int ans = bMatch();
	cout << ans << '\n';

	return 0;
}

void input() {
	cin >> R >> C;
	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			cin >> arr[y][x];
		}
	}
}

void graph_modeling() {
	int lCnt = 0;
	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			if (arr[y][x] == 0) labelX[y][x] = lCnt;
			else if (arr[y][x] == 2) labelX[y][x] = ++lCnt;
		}
		lCnt++;
	}

	lCnt = 0;
	for (int x = 0; x < C; x++) {
		for (int y = 0; y < R; y++) {
			if (arr[y][x] == 0) labelY[y][x] = lCnt;
			else if (arr[y][x] == 2) labelY[y][x] = ++lCnt;
		}
		lCnt++;
	}

	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			if (arr[y][x] == 0) {
				adj[labelX[y][x]].push_back(labelY[y][x]);
			}
		}
	}
}

int bMatch() {
	int ret = 0;
	memset(A, -1, sizeof(A));
	memset(B, -1, sizeof(B));
	for (int i = 0; i < MAXV + 5; i++) {
		memset(visited, 0, sizeof(visited));
		if (dfs(i)) ret++;
	}
	return ret;
}

bool dfs(int a) {
	visited[a] = true;
	for (int b : adj[a]) {
		if (B[b] == -1 || (!visited[B[b]] && dfs(B[b]))) {
			A[a] = b;
			B[b] = a;
			return true;
		}
	}
	return false;
}