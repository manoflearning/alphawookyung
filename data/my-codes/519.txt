#include <stdio.h>
#include <math.h>

int endnumber_count(int N) {
	int i, l;
	int d_count, flag;
	int N_count=0;
	
	for(i=666; ; i++) {
		for(d_count=2; ; d_count++)
			if(i/(int)pow(10, d_count)==0) break;
		
		flag=0;
		for(l=1; l<=d_count; l++) {
			if(i%(int)pow(10, l)/(int)pow(10, l-1)==6) flag++;
			else {
				flag=0;
				if(d_count-l<3) break; 
			}
			if(flag==3) {
				N_count++;
				break;
			}
		}
		
		if(N_count==N) break;
	}
	
	return i;
}

int main (void) {
	int N;
		
	scanf("%d", &N);
	
	printf("%d", endnumber_count(N));
	
	return 0;
}