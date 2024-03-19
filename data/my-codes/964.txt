#include <iostream>

int computer_count(int multi[], int n);

int main(void) {
	int n;
	int multi[500000];
	
	std::cin>>n;
	
	for(int i=0; i<n; i++)
		std::cin>>multi[i];
		
	std::cout<<computer_count(multi, n);
	
	return 0;
}

int computer_count(int multi[], int n) {
	int count=0;
	
	count+=multi[0];
	for(int i=1; i<n; i++)
		count+=multi[i]-1;
	
	return count;
}
