#include <iostream>

int chocolate(int n, int m);

int main(void) {
	int n, m;
	
	std::cin>>n>>m;
	
	std::cout<<chocolate(n, m);
	
	return 0;
}

int chocolate(int n, int m) {
	return n*m-1;
}
