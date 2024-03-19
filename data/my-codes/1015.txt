#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;

int dp[505][505];

int E(int N, int K);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	memset(dp, -1, sizeof(dp));

	int N, K;
	cin >> N >> K;

	cout << E(N, K);

	return 0;
}

int E(int N, int K) {
	int& ret = dp[N][K];
	if (ret != -1) return ret;
	if (N == 0 || K == 1) return ret = N;

	ret = 1e9 + 7;
	for (int mid = 1; mid <= N; mid++) {
		ret = min(ret, max(E(mid - 1, K - 1), E(N - mid, K)) + 1);
	}
	return ret;
}