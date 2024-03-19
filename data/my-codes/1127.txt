#include <iostream>
#include <algorithm>

int dp[1001];
int card[10001];

int max_price(int n);

int main() {
	int n;
	std::cin>>n;
	
	for(int i=1; i<=n; i++)
		std::cin>>card[i];
		
	std::cout<<max_price(n);
	
	return 0;
}

int max_price(int n) {
	if(n==0) return 0;
	if(dp[n]!=0) return dp[n];
	
	for(int i=1; i<=n; i++)
		dp[n] = std::max(dp[n], card[i] + max_price(n-i));
	
	return dp[n];
}