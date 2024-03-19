#include <iostream>

int main(void) {
	char chess[8][9];
	for(int i=0; i<8; i++)
		std::cin>>chess[i];
		
	int count=0;
	for(int i=0; i<8; i++)
		for(int l=i%2; l<8; l+=2)
			if(chess[i][l]=='F')
				count++;
			
	std::cout<<count;
		
	return 0;
}