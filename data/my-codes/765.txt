#include <iostream>
#include <cmath>

int dot_count(int n);

int main(void) {
	int n;
	
	std::cin>>n;
	
	std::cout<<dot_count(n);
	
	return 0;
}

int dot_count(int n) {
	int count=2;
	
	for(int i=0; i<n; i++) {
		count+=count-1;
	}
	
	return pow(count, 2);
}