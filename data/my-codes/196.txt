#include <iostream>
#include <vector>
#include <cstring>
#include <queue>
using namespace std;

const int MAXV = 1e6;

vector<int> adj[MAXV + 5];
int prv[MAXV + 5];

void bfs(int st);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	for (int x = 2; x <= MAXV; x++) {
		if (x % 3 == 0) adj[x].push_back(x / 3);
		if (x % 2 == 0) adj[x].push_back(x / 2);
		adj[x].push_back(x - 1);
	}

	int N; cin >> N;

	bfs(N);

	vector<int> ans;
	for (int i = 1; i != 0; i = prv[i]) {
		ans.push_back(i);
	}

	cout << ans.size() - 1 << '\n';
	while (!ans.empty()) {
		cout << ans.back() << ' ';
		ans.pop_back();
	}

	return 0;
}

void bfs(int st) {
	memset(prv, -1, sizeof(prv));

	queue<int> q;
	q.push(st);
	prv[st] = 0;

	while (!q.empty()) {
		int now = q.front();
		q.pop();

		for (int next : adj[now]) {
			if (prv[next] == -1) {
				prv[next] = now;
				q.push(next);
			}
		}
	}
}