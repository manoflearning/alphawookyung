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

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

const int MAX = 2e5 + 5;

int n, q, visited[MAX];
vi adj[MAX];

void input() {
	cin >> n >> q;
	for (int i = 0; i < q; i++) {
		int l, r;
		cin >> l >> r;
		adj[l - 1].push_back(r);
		adj[r].push_back(l - 1);
	}
}

void dfs(int v) {
	visited[v] = 1;

	for (auto& i : adj[v]) {
		if (!visited[i]) dfs(i);
	}
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	dfs(0);

	if (visited[n]) cout << "Yes";
	else cout << "No";

	return 0;
}