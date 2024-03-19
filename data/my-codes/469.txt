#include <iostream>

int bar_count(int x);

int main(void) {
	int x;
	
	std::cin>>x;
	
	std::cout<<bar_count(x);
	
	return 0;
}

int bar_count(int x) {
	int f_x=64;
	int count=0;
	
	while(x>0) {
		if(f_x>x) f_x/=2;
		else {
			x-=f_x;
			count++;
		}
	}
	
	return count;
}
