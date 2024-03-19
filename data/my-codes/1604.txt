#include <iostream>
#include <cstring>
#include <vector>
#define ll long long
using namespace std;

const int MOD = 1e9 + 9;
const int MAX = 1e6;

int dp[MAX + 5];

int f(int n);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	memset(dp, -1, sizeof(dp));

	int T; cin >> T;

	while (T--) {
		int n; cin >> n;
		
		cout << f(n) << '\n';
	}

	return 0;
}

int f(int n) {
	int& ret = dp[n];
	if (ret != -1) return ret;
	if (n == 0) return ret = 1;

	ret = 0;
	if (n >= 1) ret = (ret + f(n - 1)) % MOD;
	if (n >= 2) ret = (ret + f(n - 2)) % MOD;
	if (n >= 3) ret = (ret + f(n - 3)) % MOD;

	return ret;
}