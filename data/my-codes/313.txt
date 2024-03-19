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

const int MAX = 709;

int six[MAX + 5];
int dp[(int)1e6 + 1], a = 6;

void init() {
	for (int i = 1; i <= MAX; i++) {
		six[i] = six[i - 1] + 4 * i - 3;
	}

	memset(dp, -1, sizeof(dp));
}

int bs(int n) {
	int l = 1, r = MAX;
	while (l < r) {
		int mid = (l + r) / 2;

		if (n < six[mid]) r = mid;
		else l = mid + 1;
	}
	return l;
}

int f(int n) {
	int& ret = dp[n];
	if (ret != -1) return ret;
	if (n == 0) return ret = 0;

	ret = n;
	
	int l = bs(n / 5), r = bs(n);
	
	for (int i = l; i < r; i++) {
		int v = n - six[i];
		
		if (f(v) >= a) continue;

		ret = min(ret, f(v) + 1);
	}
	
	return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	init();

	int n; cin >> n;

	if (n == 11 || n == 26) cout << 6;
	else {
		if (n > 146858) a = 3;
		else if (n > 130) a = 4;
		
		cout << f(n);
	}

	return 0;
}