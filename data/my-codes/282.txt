#include <iostream>

double sp_factorial(int x, int y);

int main()
{
	int t;
	std::cin>>t;
	
	for(int i=0; i<t; i++)
	{
		int n, m;
		std::cin>>n>>m;
		
		printf("%.0lf\n", sp_factorial(m, n));
	}
	
	return 0;
}

double sp_factorial(int x, int y)
{
	double ans=1;
	for(float i=x-y+1; i<=x; i++)
	{
		ans *= i;
		ans /= (x-i+1);
	}
	
	return ans;
}