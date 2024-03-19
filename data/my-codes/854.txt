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

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N, M;
	cin >> N >> M;

	vector<int> a(M);
	for (int& i : a) cin >> i;

	int l = 1, r = 1e9;
	while (l < r) {
		int mid = (l + r) >> 1;
		int cnt = 0;

		for (int& i : a)
			cnt += (i + mid - 1) / mid;

		if (cnt <= N) r = mid;
		else l = mid + 1;
	}

	cout << l;

	return 0;
}