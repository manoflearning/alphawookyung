#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int INF = 1e9 + 7;

int dp[55][1005];

int f(int B, int M);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	memset(dp, -1, sizeof(dp));

	int tc; cin >> tc;

	while (tc--) {
		int B, M;
		cin >> B >> M;

		cout << f(B, M) << '\n';
	}

	return 0;
}

int f(int B, int M) {
	int& ret = dp[B][M];
	if (ret != -1) return ret;
	if (B == 1 || M == 0) return ret = M;

	ret = INF;
	for (int mid = 1; mid <= M; mid++) {
		ret = min(ret, max(f(B - 1, mid - 1) + 1, f(B, M - mid) + 1));
	}

	return ret;
}