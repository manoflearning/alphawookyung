#include <iostream>

int main(void) {
	int t;
	
	std::cin>>t;
	
	for(int i=0; i<t; i++) {
		int n, k;
		
		std::cin>>n>>k;
		
		int candy;
		int count=0;
		
		for(int l=0; l<n; l++) {
			std::cin>>candy;
			
			count+=candy/k;
		}
		
		std::cout<<count<<std::endl;
	}
	
	return 0;
}