#include <iostream>
#include <vector>
#include <queue>
#include <cstring>
using namespace std;

const int MAX = 1e4;

vector<int> adj[MAX + 5];
int prv[MAX + 5], com[MAX + 5];

void graph_modeling();
void bfs(int st, int en);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	graph_modeling();

	int T; cin >> T;

	while (T--) {
		int st, en;
		cin >> st >> en;

		bfs(st, en);

		string ans;
		for (int i = en; i != st; i = prv[i]) {
			if (com[i] == 0) ans.push_back('D');
			else if (com[i] == 1) ans.push_back('S');
			else if (com[i] == 2) ans.push_back('L');
			else if (com[i] == 3) ans.push_back('R');
		}

		for (int i = ans.size() - 1; i >= 0; i--)
			cout << ans[i];
		cout << '\n';
	}

	return 0;
}

void graph_modeling() {
	for (int i = 0; i <= 9999; i++) {
		adj[i].push_back(2 * i % MAX);
		adj[i].push_back((i + 9999) % MAX);
		adj[i].push_back(i * 10 % MAX + i / 1000);
		adj[i].push_back(i / 10 + i % 10 * 1000);
	}
}

void bfs(int st, int en) {
	memset(prv, -1, sizeof(prv));
	
	queue<int> q;
	q.push(st);
	prv[st] = 0;

	while (!q.empty()) {
		int now = q.front();
		q.pop();

		for (int i = 0; i < 4; i++) {
			int next = adj[now][i];
			if (prv[next] == -1) {
				prv[next] = now;
				com[next] = i;
				q.push(next);
			}
		}
	}
}