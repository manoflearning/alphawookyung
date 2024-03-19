#include <stdio.h>

int i, l, j;

int bongzi_count(int N) {
	int N5;
	int count;
	
	N5=N/5;
	for(i=N5; i>=0; i--)
		if((N-i*5)%3==0) {
			count=i+(N-i*5)/3;
			break;
		}
		
	if(i==-1) return -1;
	else return count;
}

int main (void) {
	int N;
	
	scanf("%d", &N);
	
	printf("%d", bongzi_count(N));
	
	return 0;
}