#include <iostream>

int main() {
	int n, f;
	std::cin>>n>>f;
	
	n/=100;
	n*=100;
	
	int sam=0;
	while(n>sam) sam+=f;
	
	sam%=100;
	while(sam>=0) sam-=f;
	sam+=f;
	
	if(sam<10) 
		std::cout<<'0'<<sam;
	else
		std::cout<<sam;
		
	return 0;
}
