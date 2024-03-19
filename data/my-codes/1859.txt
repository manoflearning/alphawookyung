#include <iostream>
#include <cmath>

int chain(long long n);
long long cmax(int i);

int main() {
	long long n;
	std::cin>>n;
	
	std::cout<<chain(n);
	
	return 0;
}

int chain(long long n) {
	for(int i=1; ; i++)
		if(n <= cmax(i)) return i;
}
long long cmax(int i) {
	return (i+1) * (long long)pow(2, i+1) - 1;
}