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
const int MAXN = 1e5;

struct xy {
	int x, y;

	bool operator<(xy& rhs) {
		if (x != rhs.x) return x < rhs.x;
		return y < rhs.y;
	}
};

int N;
vector<xy> p;
int minl[MAXN + 5], maxr[MAXN + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;
	p.resize(N);
	for (auto& i : p) cin >> i.x >> i.y;

	sort(p.begin(), p.end());

	minl[0] = p[0].y;
	maxr[N - 1] = p[N - 1].y;
	for (int i = 1; i < N; i++) minl[i] = min(minl[i - 1], p[i].y);
	for (int i = N - 2; i >= 0; i--) maxr[i] = max(maxr[i + 1], p[i].y);

	int res = 1;
	for (int i = 0; i < N - 1; i++)
		if (minl[i] > maxr[i + 1]) res++;
	cout << res;

	return 0;
}