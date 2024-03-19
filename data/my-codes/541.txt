#include <iostream>

int at_least_most_multiple(int nn[]);

int main(void) {
	int nn[5];
	
	for(int i=0; i<5; i++)
		std::cin>>nn[i];
		
	std::cout<<at_least_most_multiple(nn);
	
	return 0;
}

int at_least_most_multiple(int nn[]) {
	int almm;
	
	for(int i=0; i<5; i++) {
		int flag=0;
		for(int l=0; l<5; l++)
			if(nn[i]>nn[l]) flag++;
			
		if(flag==2) {
			almm = nn[i];
			break;
		}
	}
	
	while(1) {
		int count=0;
		
		for(int i=0; i<5; i++)
			if(almm%nn[i] == 0) count++;
			
		if(count>=3) break;
		
		almm++;
	}
	
	return almm;
}
