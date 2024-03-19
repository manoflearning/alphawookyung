#include <iostream>

void chess(int r, int c, int a, int b);

int main(void) {
	int r, c;
	int a, b;
	
	std::cin>>r>>c>>a>>b;
	
	chess(r, c, a, b);
	
	return 0;
}

void chess(int r, int c, int a, int b) {
	for(int i=0; i<r; i++)
		for(int l=0; l<a; l++) {
			for(int j=0; j<c; j++) {
				if((j%2==0 && i%2==0) || (j%2==1 && i%2==1))
					for(int k=0; k<b; k++)
						std::cout<<'X';
				
				else
					for(int k=0; k<b; k++)
						std::cout<<'.';
			}
			std::cout<<std::endl;
		}
}