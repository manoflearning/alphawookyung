#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vl vector<ll>
#define vvi vector<vector<int>>
#define vvl vector<vector<ll>>
using namespace std;

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const int MAX = 5 * 1e5;

int N, label[MAX + 5];
vi adj[MAX + 5];
bool visited[MAX + 5];

bool cmp(int a, int b);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int M;
	cin >> N >> M;

	for (int i = 0; i < M; i++) {
		int u, v;
		cin >> u >> v;
		adj[u].push_back(v);
		adj[v].push_back(u);
	}

	for (int i = 1; i <= N; i++) {
		cin >> label[i];
	}

	for (int u = 1; u <= N; u++) {
		sort(adj[u].begin(), adj[u].end(), cmp);

		int flag = 1;
		for (int v : adj[u])
			if (flag == label[v]) flag++;

		if (flag != label[u]) {
			cout << -1 << '\n';
			return 0;
		}
	}

	vector<pii> arr;
	for (int i = 1; i <= N; i++) {
		arr.push_back({ label[i], i });
	}

	vector<int> ans;
	
	sort(arr.begin(), arr.end());
	for (int i = 0; i < arr.size(); i++) ans.push_back(arr[i].second);

	for (int i = 0; i < N; i++)
		cout << ans[i] << ' ';

	return 0;
}

bool cmp(int a, int b) {
	return label[a] < label[b];
}