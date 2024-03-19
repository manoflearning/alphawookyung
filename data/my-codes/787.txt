#include <iostream>
#include <algorithm>
#include <vector>

std::vector<int> dp(10001, 10001);
std::vector<int> coin;

int main() {
	int n, k;
	std::cin>>n>>k;
	
	coin.resize(n);
	for(int i=0; i<n; i++)
		std::cin>>coin[i];
	
	dp[0] = 0;
	for(int i=1; i<=k; i++) {
		for(int l=0; l<n; l++) {
			if(i-coin[l]>=0 && dp[i-coin[l]] != -1)
				dp[i] = std::min(dp[i], dp[i-coin[l]]+1);
		}
	}
	
	if(dp[k]<=10000) std::cout<<dp[k];
	else std::cout<<"-1";
	
	return 0;
}
