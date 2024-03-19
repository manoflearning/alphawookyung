#include <iostream>

int maximum_count(int a, int b, int c);

int main(void) {
	int a, b, c;
	
	std::cin>>a>>b>>c;
	
	std::cout<<maximum_count(a, b, c);
	
	return 0;
}

int maximum_count(int a, int b, int c) {
	int max;
	
	max = (b-a > c-b) ? b-a : c-b;
	
	return max-1;
}