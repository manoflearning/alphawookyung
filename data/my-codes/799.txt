#include <iostream>
#include <cstring>

int min(int a, int b);
int max(int a, int b);

int main(void) {
	int a, b;
	
	std::cin>>a>>b;
	
	std::cout<<min(a, b)<<' '<<max(a, b);
	
	return 0;
}

int min(int a, int b) {
	for(int i=1; i<=a || i<=b; i*=10) {
		if(a%(i*10)/i == 6) a-=i;
		if(b%(i*10)/i == 6) b-=i;
	}
	
	return a+b;
}
int max(int a, int b) {
	for(int i=1; i<=a || i<=b; i*=10) {
		if(a%(i*10)/i == 5) a+=i;
		if(b%(i*10)/i == 5) b+=i;
	}
	
	return a+b;
}