#include <iostream>

#define chicken(l, t) (l)*2+(t-l) 

int main(void) {
	int m;
	
	std::cin>>m;
	
	for(int i=0; i<m; i++) {
		int u, t;
		
		std::cin>>u>>t;
		
		for(int l=0; l<=t; l++)
			if(chicken(l, t) == u) {
				std::cout<<t-l<<' '<<l<<std::endl;
				break;
			}
	}
	
	return 0;
}