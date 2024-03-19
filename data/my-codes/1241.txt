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
const int MAXV = 2500;

vector<int> adj[MAXV + 5];
int A[MAXV + 5], dep[MAXV + 5];

void dfs(int now, int prv);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int N; cin >> N;
	for (int i = 1; i <= N; i++) cin >> A[i];
	for (int i = 0; i < N - 1; i++) {
		int u, v;
		cin >> u >> v;
		adj[u].push_back(v);
		adj[v].push_back(u);
	}

	dep[0] = -1;
	dfs(1, 0);

	int s0 = 0, s1 = 0, n0 = 0, n1 = 0;
	for (int i = 1; i <= N; i++) {
		if (dep[i] & 1) s1 += A[i], n1++;
		else s0 += A[i], n0++;
	}

	if (s0 % 12 == s1 % 12) cout << N;
	else if ((s0 + 1) % 12 == s1 % 12) cout << n1;
	else if (s0 % 12 == (s1 + 1) % 12) cout << n0;
	else cout << 0;

	return 0;
}

void dfs(int now, int prv) {
	dep[now] = dep[prv] + 1;
	for (int i : adj[now])
		if (i != prv) dfs(i, now);
}