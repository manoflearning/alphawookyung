#include <iostream>

int dp[1001];

int tile(int n);

int main() {
	int n;
	std::cin>>n;
	
	std::cout<<tile(n);
		
	return 0;
}

int tile(int n) {
	if(dp[n]!=0) return dp[n];
	if(n<0) return 0;
	if(n==0) return 1;
	
	dp[n] += tile(n-1) + 2*tile(n-2);
	
	dp[n] %= 10007;
	
	return dp[n];
}