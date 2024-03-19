#include <iostream>
#include <algorithm>

int dp[100001];

int min_nul(int n);

int min_nul_1(int n);

int main() {
	int n;
	std::cin>>n;
	
	std::cout<<min_nul(n);
	
	return 0;
}

int min_nul(int n) {
	if(dp[n]!=0) return dp[n];
	
	min_nul_1(n);
	
	return dp[n]; 
}

int min_nul_1(int n) {
	int min = n;
	
	for(int i=1; i*i<=n; i++) {
		int sam = n - i*i;
		if(dp[sam]!=0)
			min = std::min(min, dp[sam]+1);
		else
			min = std::min(min, min_nul_1(sam)+1);
	}
	dp[n] = min;
	
	return dp[n];
}
