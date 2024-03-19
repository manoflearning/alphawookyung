#include <iostream>

int station_number(int k);

int main(void) {
	int n;
	
	std::cin>>n;
	
	for(int i=0; i<n; i++) {
		int k;
		
		std::cin>>k;
		
		std::cout<<station_number(k)<<std::endl;
	}
	
	return 0;
}

int station_number(int k) {
	double pas=0;
	
	for(int i=0; i<k; i++) {
		pas+=0.5;
		pas*=2;
	}
	
	return pas;
}