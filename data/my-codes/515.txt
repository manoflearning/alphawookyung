#include <iostream>
#include <algorithm>

int main() {
	int n;
	std::cin>>n;
	
	int a[n], b[n];
	for(int i=0; i<n; i++)
		std::cin>>a[i];
	for(int i=0; i<n; i++)
		std::cin>>b[i];
	
	std::sort(a, a+n);
	std::sort(b, b+n);
	
	int max=0;
	for(int i=0; i<n; i++)
		max += a[i]*b[n-1-i];
		
	std::cout<<max;
	
	return 0;
}
