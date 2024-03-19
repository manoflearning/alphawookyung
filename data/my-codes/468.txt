#include <iostream>
#include <math.h>

int find_round(int n, int kim, int lim);

int main(void) {
	int n, kim, lim;
	
	std::cin>>n>>kim>>lim;
	
	std::cout<<find_round(n, kim, lim);
	
	return 0;
}

int find_round(int n, int kim, int lim) {
	int count=0;
	int f_n;
	
	for(int i=2; i<=n; i*=2)
		f_n = i;
	
	for(int i=2; i<=f_n*2; i*=2) {
		count++;
		for(int l=0; l<=f_n*2; l+=i) {
			if(l<kim && kim<=l+i && l<lim && lim<=l+i) {
				i=f_n*2+1;
				break;
			}
		}
	}
	
	return count;
}
