#include <iostream>
#include <cstring>

void decoding(char public_key1[][11], char public_key2[][11], int decode[], int n);

int main(void) {
	char public_key1[1000][11], public_key2[1000][11];
	char code[1000][11];
	int decode[1000];
	int t;
	
	std::cin>>t;
	
	for(int i=0; i<t; i++) {
		int n;
		
		std::cin>>n;
		
		for(int l=0; l<n; l++)
			std::cin>>public_key1[l];
		for(int l=0; l<n; l++)
			std::cin>>public_key2[l];
		for(int l=0; l<n; l++)
			std::cin>>code[l];
			
		decoding(public_key1, public_key2, decode, n);
		
		for(int l=0; l<n; l++)
			std::cout<<code[decode[l]]<<' ';
		std::cout<<std::endl;
	}
	
	return 0;
}

void decoding(char public_key1[][11], char public_key2[][11], int decode[], int n) {
	for(int i=0; i<n; i++)
		for(int l=0; l<n; l++)
			if(strcmp(public_key2[i], public_key1[l]) == 0)
				decode[l] = i;
}
