#include <iostream>

#define MAX 1000 1000

int dp[1001][1001];

int main() {
	int n, k;
	std::cin>>n>>k;
	
	dp[1][0] = 1, dp[1][1] = 1;
	for(int i=2; i<=n; i++) {
		dp[i][0] = 1;
		for(int l=1; l<=i && l<=k; l++)
			dp[i][l] = ( dp[i-1][l-1] + dp[i-1][l] ) % 10007;
	}
	std::cout<<dp[n][k];
	
	return 0;
}