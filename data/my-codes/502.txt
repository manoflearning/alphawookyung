#include <iostream>

void candy_count(int n, int candy[], int sum);

int main(void) {
	int n;
	int candy[999];
	int sum=0;
	
	std::cin>>n;
	
	for(int i=0; i<n; i++) {
		std::cin>>candy[i];
		
		sum+=candy[i];
	}
	
	candy_count(n, candy, sum);
	
	return 0;
}

void candy_count(int n, int candy[], int sum) {
	int aver=sum/2;
	int indi_candy[999];
	int ic;
		
	for(int i=0; i<n; i++) {
		ic = aver;
		for(int l=i%2; l<i; l+=2)
			ic-=candy[l];
		for(int l=i+1; l<n; l+=2)
			ic-=candy[l];
		
		indi_candy[i] = ic;
	}
	
	for(int i=0; i<n; i++)
		std::cout<<indi_candy[i]<<std::endl;
}
