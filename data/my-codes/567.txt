#include <iostream>
#include <cmath>

void find_fullsquarenumber(int m, int n);

int main(void) {
	int m, n;
	
	std::cin>>m>>n;
	
	find_fullsquarenumber(m, n);
	
	return 0;
}

void find_fullsquarenumber(int m, int n) {
	int flag=0;
	int count=0;
	int min=n;
	int c=1;
	
	while(1) {
		if(m<=pow(c, 2) && pow(c,2) <=n) {
			count+=pow(c, 2);
			
			if(flag==0) min=pow(c, 2);
			
			flag++;
		}
		
		if(n<=pow(c, 2)) break;
		
		c++;
	}
	if(flag==0) std::cout<<"-1";
	else {
		std::cout<<count<<std::endl<<min;
	}
}
