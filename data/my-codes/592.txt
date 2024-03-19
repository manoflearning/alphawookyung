#include <iostream>

void perfect_number(int n);

int main(void) {
	int n;
	
	while(1) {
		std::cin>>n;
		
		if(n==-1) break;
		
		perfect_number(n);
	}
	
	return 0;
}

void perfect_number(int n) {
	int weaknumber[100000];
	int count=0;
	int sum=0;
	
	for(int i=1; i<n; i++)
		if(n%i==0) {
			weaknumber[count] = i;
			sum += i;
			count++;
		}
	
	if(n == sum) {
		std::cout<<n<<" = ";
		
		std::cout<<weaknumber[0];
		for(int i=1; i<count; i++)
			std::cout<<" + "<<weaknumber[i];
		
		std::cout<<std::endl;
	}
	else
		std::cout<<n<<" is NOT perfect."<<std::endl;
}
