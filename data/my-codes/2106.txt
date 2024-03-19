#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
using namespace std;

const int INF = 1e9 + 7;

int N, arr[1005][3];
int dp[1005][3], st;

int f(int x, int d);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	cin >> N;

	for (int i = 1; i <= N; i++) {
		cin >> arr[i][0] >> arr[i][1] >> arr[i][2];
	}

	int ans = INF;
	for (st = 0; st < 3; st++) {
		memset(dp, -1, sizeof(dp));
		ans = min(ans, f(1, st));
	}
	cout << ans;

	return 0;
}

int f(int x, int d) {
	int& ret = dp[x][d];
	if (ret != -1) return ret;
	if (x == N - 1) {
		ret = INF;
		for (int i = 0; i < 3; i++)
			if (i != d && i != st) ret = min(ret, f(x + 1, i));
		return ret += arr[x][d];
	}

	ret = INF;
	for (int i = 0; i < 3; i++)
		if (i != d) ret = min(ret, f(x + 1, i));

	return ret += arr[x][d];
}