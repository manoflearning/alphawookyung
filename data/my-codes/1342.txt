#include <iostream>

int main(void) {
	int t;
	
	std::cin>>t;
	
	for(int i=0; i<t; i++) {
		int v, e;
		
		std::cin>>v>>e;
		
		std::cout<<e-v+2<<std::endl;
	}
	
	return 0;
}