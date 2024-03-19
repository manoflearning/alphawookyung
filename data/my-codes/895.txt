#include <iostream>

int count(int n);
int from_1_to_n(int n);

int main(void) {
	int N;
	
	std::cin>>N;
	
	std::cout<<count(N);
	
	return 0;
}

int count(int N) {
	int count=1;
	int flag = N-from_1_to_n(2);
	
	for(int i=2; flag>0; i++, flag = N-from_1_to_n(i))
		if(flag%i==0)
			count++;
	
	return count;
}

int from_1_to_n(int n) {
	if(n>1) return (n-1) + from_1_to_n(n-1);
	else return 0;
}
