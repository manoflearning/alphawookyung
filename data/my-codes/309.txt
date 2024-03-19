#include <iostream>
#include <algorithm>

#define RED 0
#define GREEN 1
#define BLUE 2
#define MAX 1000

int dp[MAX+1][3];

int main() {
	int n;
	std::cin>>n;
	
	int color[n+1][3];
	for(int i=1; i<=n; i++)
	for(int l=0; l<3; l++)
		std::cin>>color[i][l];
		
	dp[1][RED] = color[1][RED];
	dp[1][GREEN] = color[1][GREEN];
	dp[1][BLUE] = color[1][BLUE];
	for(int i=2; i<=n; i++) {
		dp[i][RED] = std::min(dp[i-1][GREEN], dp[i-1][BLUE]);
		dp[i][RED] += color[i][RED];
		dp[i][GREEN] = std::min(dp[i-1][RED], dp[i-1][BLUE]);
		dp[i][GREEN] += color[i][GREEN];
		dp[i][BLUE] = std::min(dp[i-1][RED], dp[i-1][GREEN]);
		dp[i][BLUE] += color[i][BLUE];
	}
	
	std::cout<<std::min(dp[n][RED], std::min(dp[n][GREEN], dp[n][BLUE]));
	
	return 0;
}
