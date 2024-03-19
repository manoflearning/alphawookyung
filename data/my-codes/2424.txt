#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const int MAXV = 2e5;

int n;
vector<ll> a, b;
vector<int> l, r;
int indeg[MAXV + 5], outdeg[MAXV + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> n;
	a.resize(n + 1), b.resize(n + 1);
	for (int i = 1; i <= n; i++) cin >> a[i];
	for (int i = 1; i <= n; i++) cin >> b[i];

	for (int i = 1; i <= n; i++) {
		if (b[i] != -1) {
			indeg[b[i]]++;
			outdeg[i]++;
		}
	}

	queue<int> q;

	for (int i = 1; i <= n; i++)
		if (indeg[i] == 0) q.push(i);
	
	ll res = 0;
	for (int i = 0; i < n; i++) {
		int v = q.front();
		q.pop();

		res += a[v];

		if (b[v] == -1) {
			l.push_back(v);
			continue;
		}

		if (a[v] >= 0) {
			a[b[v]] += a[v];
			l.push_back(v);
		}
		else r.push_back(v);

		indeg[b[v]]--;
		if (indeg[b[v]] == 0) q.push(b[v]);
	}

	cout << res << '\n';
	for (int i = 0; i < l.size(); i++) cout << l[i] << ' ';
	for (int i = r.size() - 1; i >= 0; i--) cout << r[i] << ' ';

	return 0;
}