#include <stdio.h>

int main (void) {
	int i;
	int N, ascending[2000001]={0,}, f;
	
	scanf("%d", &N);
	
	for(i=0; i<N; i++) {
		scanf("%d", &f);
		ascending[f+1000000]=1;
	}
	
	for(i=0; i<=2000000; i++)
		if(ascending[i])
			printf("%d\n", i-1000000);
	
	return 0;
}