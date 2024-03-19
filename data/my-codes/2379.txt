#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const int d[] = { -1, 1 };
const int MAX = 1e5;

int N;
vector<int> arr;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;

	arr.resize(N);
	for (int& i : arr) cin >> i;

	ll ans = 0;
	for (int mx = 0; mx <= 30; mx++) {
		ll cur = 0, best = 0;
		for (int i = 0; i < N; i++) {
			int val = (arr[i] > mx ? -INF : arr[i]);
			cur += val;
			best = min(best, cur);
			ans = max(ans, (cur - best) - mx);
		}
	}

	cout << ans << '\n';

	return 0;
}