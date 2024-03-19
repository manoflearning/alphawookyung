#include <iostream>

#define LCOUNT 0
#define RCOUNT 1

#define MAX 1000

int dp[MAX+1][2];

int bitonic_LCOUNT(int bitonic[], int flag);
int bitonic_RCOUNT(int bitonic[], int flag, int n);

int main()
{
	int n;
	std::cin>>n;
	int bitonic[n+1];
	for(int i=1; i<=n; i++)
		std::cin>>bitonic[i];
		
	int max=0;
	for(int i=1; i<=n; i++)
		max = std::max(max, bitonic_LCOUNT(bitonic, i) + bitonic_RCOUNT(bitonic, i, n) - 1);
		
	std::cout<<max;
	
	return 0;
}

int bitonic_LCOUNT(int bitonic[], int flag)
{
	if(dp[flag][LCOUNT]!=0) return dp[flag][LCOUNT];
	
	for(int i=1; i<flag; i++)
		if(bitonic[i]<bitonic[flag])
			dp[flag][LCOUNT] = std::max(dp[flag][LCOUNT], bitonic_LCOUNT(bitonic, i));
	dp[flag][LCOUNT]++;
	
	return dp[flag][LCOUNT];
}

int bitonic_RCOUNT(int bitonic[], int flag, int n)
{
	if(dp[flag][RCOUNT]!=0) return dp[flag][RCOUNT];
	
	for(int i=flag+1; i<=n; i++)
		if(bitonic[flag]>bitonic[i])
			dp[flag][RCOUNT] = std::max(dp[flag][RCOUNT], bitonic_RCOUNT(bitonic, i, n));
	dp[flag][RCOUNT]++;
	
	return dp[flag][RCOUNT];
}