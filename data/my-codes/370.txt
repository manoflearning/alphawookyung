#include <iostream>
#include <vector>
#include <algorithm>

int main() {
	int x;
	
	std::cin>>x;
	
	std::vector<int> dp(x+1);
	dp[1]=0, dp[2]=1, dp[3]=1;
	
	for(int i=4; i<=x; i++) {
		dp[i] = dp[i-1]+1;
		if(i%3==0) dp[i] = std::min(dp[i], dp[i/3]+1);
		if(i%2==0) dp[i] = std::min(dp[i], dp[i/2]+1);
	}
	
	std::cout<<dp[x];
	
	return 0;
}
