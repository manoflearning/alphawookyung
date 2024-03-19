#include <iostream>

#define MAX 100

long long dp[MAX+1];

int main()
{
	int t;
	std::cin>>t;
	
	for(int i=0; i<t; i++)
	{
		long long n;
		std::cin>>n;
		
		dp[1] = 1;
		dp[2] = 1;
		dp[3] = 1;
		dp[4] = 2;
		dp[5] = 2;
		for(int l=6; l<=n; l++)
			dp[l] = dp[l-1] + dp[l-5];
		
		std::cout<<dp[n]<<'\n';
	}
	
	return 0;
}