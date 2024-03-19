#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>
#define vs vector<string>
#define vvs vector<vector<string>>
#define fr first
#define sc second

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1, 1, 1, -1, -1 };
const int dx[] = { 1, -1, 0, 0, 1, -1, 1, -1 };

int n, a[1000005], visited[100005], cnt;
vi adj[100005], ans;

void input() {
	cin >> n;
	for (int i = 0; i < n - 1; i++) {
		int u, v;
		cin >> u >> v;
		adj[u].push_back(v);
		adj[v].push_back(u);
	}

	ans.resize(n);
	for (auto& i : ans) cin >> i;
}

int f() {
	queue<int> q;

	q.push(1);
	visited[1] = 1;
	cnt++;

	for (int i = 0; i < n; i++) {
		int v = q.front();
		q.pop();
		
		if (ans[i] != v) return 0;

		set<int> b;
		for (auto& j : adj[v]) {
			if (visited[j]) continue;
			b.insert(j);
		}
		for (int j = cnt; j < cnt + b.size(); j++) {
			if (b.find(ans[j]) == b.end()) return 0;

			q.push(ans[j]);
			visited[ans[j]] = 1;
		}

		cnt += b.size();
	}

	return 1;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	cout << f();

	return 0;
}
