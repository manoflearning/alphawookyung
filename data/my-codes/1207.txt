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
const int MAX = 1e5;

int N, M, cnt, ans[MAX + 5];
pii c[MAX + 5];
int a[MAX + 5];

void f(int idx, int now);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	memset(a, -1, sizeof(a));

	cin >> N >> M;
	for (int i = 1; i <= N; i++)
		cin >> c[i].first >> c[i].second;

	for (int i = N; i >= 1; i--) {
		f(c[i].first, i);

		ans[i] = cnt;
	}

	for (int i = 1; i <= N; i++) cout << ans[i] << '\n';

	return 0;
}

void f(int idx, int now) {
	if (a[idx] != -1 && a[idx] < now) return;

	if (a[idx] == -1) cnt++;
	else if (idx == c[a[idx]].first) f(c[a[idx]].second, a[idx]);

	a[idx] = now;
}