#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

string name[] = { "Bessie", "Buttercup", "Belinda", "Beatrice", "Bella", "Blue", "Betsy", "Sue" };
vector<int> adj[8];
bool visited[8];

void dfs(int now, int prv);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	sort(name, name + 8);

	int N; cin >> N;
	for (int i = 0; i < N; i++) {
		string u, v, s;
		cin >> u >> s >> s >> s >> s >> v;
		int uu = find(name, name + 8, u) - name, vv = find(name, name + 8, v) - name;
		adj[uu].push_back(vv);
		adj[vv].push_back(uu);
	}

	for (int i = 0; i < 8; i++) {
		if (adj[i].size() == 0) cout << name[i] << '\n';
		else if (!visited[i] && adj[i].size() == 1) dfs(i, -1);
	}
	
	return 0;
}

void dfs(int now, int prv) {
	visited[now] = true;
	cout << name[now] << '\n';

	for (int i : adj[now])
		if (i != prv) dfs(i, now);
}