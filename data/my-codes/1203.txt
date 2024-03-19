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
const int MAX = 5e4;

struct wxd {
	int w, x, d;
	bool operator<(wxd& rhs) {
		return x < rhs.x;
	}
};

int N, L, T, total;
wxd a[MAX + 5];
vector<pii> arr;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> L;
	for (int i = 0; i < N; i++) 
		cin >> a[i].w >> a[i].x >> a[i].d;

	sort(a, a + N);

	vector<int> l, r;
	for (int i = 0; i < N; i++) {
		total += a[i].w;
		if (a[i].d == -1) l.push_back(a[i].x);
		else r.push_back(a[i].x);
	}

	for (int i = 0; i < l.size(); i++) arr.push_back({ l[i], a[i].w });
	for (int i = 0; i < r.size(); i++) arr.push_back({ L - r[i], a[l.size() + i].w });

	sort(arr.begin(), arr.end());

	int sum = 0;
	for (int i = 0; i < N; i++) {
		sum += arr[i].second;
		T = arr[i].first;
		if (2 * sum >= total) break;
	}
	
	queue<int> q;
	int res = 0;
	for (int i = 0; i < N; i++) {
		if (a[i].d == -1) {
			while (!q.empty() && 2 * T < a[i].x - q.front()) q.pop();
			res += q.size();
		}
		else q.push(a[i].x);
	}
	cout << res;

	return 0;
}