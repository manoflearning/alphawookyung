#include <iostream>
#include <vector>
#include <cstring>
using namespace std;

const int MAX = 500;

int A[MAX + 5], B[MAX + 5];
vector<int> adj[MAX + 5];
bool visited[MAX + 5];
int cntX[MAX + 5], cntY[MAX + 5];

void input();
int bMatch();
bool dfs(int a);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();

	bMatch();
	
	for (int i = 1; i <= MAX; i++) {
		if (cntX[i] > 0 && A[i] == -1) return cout << "Mirko", 0;
		if (cntY[i] > 0 && B[i] == -1) return cout << "Mirko", 0;
	}

	cout << "Slavko";
	
	return 0;
}

void input() {
	int N; cin >> N;

	for (int i = 0; i < N; i++) {
		int x, y;
		cin >> x >> y;
		cntX[x]++, cntY[y]++;
		adj[x].push_back(y);
	}
}

int bMatch() {
	memset(A, -1, sizeof(A));
	memset(B, -1, sizeof(B));

	int ret = 0;
	for (int i = 1; i <= MAX; i++) {
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