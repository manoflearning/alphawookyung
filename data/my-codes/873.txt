#include <iostream>

void star_draw14(int n);

int main(void) {
	int n;
	
	std::cin>>n;
	
	star_draw14(n);
	
	return 0;
}

void star_draw14(int n) {
	for(int i=0; i<n; i++) {
		for(int l=0; l<n; l++) {
			std::cout<<'*';
		}
		std::cout<<std::endl;
	}
}
