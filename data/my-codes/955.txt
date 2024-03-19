#include <iostream>

#define MAX 90

long long dp[MAX+1];

long long fibonacci(int n);

int main() {
	int n;
	std::cin>>n;
	
	std::cout<<fibonacci(n);
	
	return 0;
}

long long fibonacci(int n) {
	if(dp[n]!=0) return dp[n];
	if(n==1) return 1;
	if(n<=0) return 0;
	
	dp[n] = fibonacci(n-1) + fibonacci(n-2);
	
	return dp[n];
}