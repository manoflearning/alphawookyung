#include <iostream>
#include <vector>
#include <algorithm>

int stair[301];
int dp[301][2];

int stair_count(int n, bool fl);

int main() {
	int n;
	std::cin>>n;
	
	for(int i=1; i<=n; i++)
		std::cin>>stair[i];
		
	std::cout<<stair_count(n, 0);
	
	return 0;
}

int stair_count(int n, bool fl) {
	if(dp[n][fl]!=0) return dp[n][fl];
	if(n<=0) return 0;
	
	dp[n][fl] = stair[n];
	if(fl)
		dp[n][fl] += stair_count(n-2, 0);
	else
		dp[n][fl] += std::max(stair_count(n-1, 1), stair_count(n-2, 0));
	
	return dp[n][fl];
}
