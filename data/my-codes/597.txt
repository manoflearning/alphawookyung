#include <iostream>
#include <algorithm>

const int MAX = 100000;

int dp[MAX][3];

int main() {
	int t;
	std::cin>>t;
	
	for(int i=0; i<t; i++) {
		int n;
		std::cin>>n;
		
		int sti[2][n];
		for(int l=0; l<2; l++)
		for(int j=0; j<n; j++)
			std::cin>>sti[l][j];
		
		dp[0][0]=0, dp[0][1]=sti[0][0], dp[0][2]=sti[1][0];
		for(int l=1; l<n; l++) {
			dp[l][0] = dp[l-1][0];
			dp[l][0] = std::max(dp[l][0], dp[l-1][1]);
			dp[l][0] = std::max(dp[l][0], dp[l-1][2]);
			
			dp[l][1] = sti[0][l] + std::max(dp[l-1][0], dp[l-1][2]);
			
			dp[l][2] = sti[1][l] + std::max(dp[l-1][0], dp[l-1][1]);
		}
		
		int ans = dp[n-1][0];
		ans = std::max(ans, dp[n-1][1]);
		ans = std::max(ans, dp[n-1][2]);
		std::cout<<ans<<'\n';
	}
	
	return 0;
}