#include <iostream>

#define MAX 1000

int dp[MAX+1][10];

int ascending_order(int n, int s);

int main() {
	int n;
	std::cin>>n;
	
	int ans=0;
	for(int i=0; i<=9; i++) {
		ans+=ascending_order(n, i);
		ans%=10007;
	}
	std::cout<<ans<<'\n';

	return 0;
}

int ascending_order(int n, int s) {
	if(n==1) return 1;
	if(dp[n][s]!=0) return dp[n][s];
	
	for(int i=0; i<=s; i++)
		dp[n][s] += ascending_order(n-1, i);
	dp[n][s]%=10007;
	
	return dp[n][s];
}