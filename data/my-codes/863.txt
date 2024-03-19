#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int MAX = 1e4;

int N, arr[MAX + 5], dp[MAX + 5][2];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;

	for (int i = 1; i <= N; i++)
		cin >> arr[i];

	dp[1][0] = arr[1], dp[2][0] = arr[2];
	dp[2][1] = arr[1] + arr[2];

	for (int i = 3; i <= N; i++) {
		dp[i][0] = max({ dp[i - 2][0], dp[i - 3][0], dp[i - 2][1], dp[i - 3][1] }) + arr[i];
		dp[i][1] = dp[i - 1][0] + arr[i];
	}
	
	cout << max({ dp[N][0], dp[N][1], dp[N - 1][0], dp[N - 1][1] }) << '\n';

	return 0;
}