#include <stdio.h>

void star_drawing(int N) {
	int i, l;
	
	for(i=N; i>0; i--) {
		for(l=1; l<=N-i; l++)
			printf(" ");
		for(l=1; l<=i; l++)
			printf("*");
			
		printf("\n");
	}
}

int main (void) {
	int N;
	
	scanf("%d", &N);
	
	star_drawing(N);
	
	return 0;
}
