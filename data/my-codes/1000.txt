#include <iostream>

int cal(int d);

int main(void) {
	int n;
	
	std::cin>>n;
	
	std::cout<<cal(n+1)*n;
	
	return 0;
}

int cal(int d) {
	int count=0;
	
	for(int i=1; i<=d; i++)
		count+=i;
		
	return count;
}