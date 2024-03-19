#include <stdio.h>

int i, l, j;

int main (void) {
	int A, B, C, D, E;
	
	scanf("%d %d %d", &A, &B, &C);
	
	D = C-B;
	
	if(D>0)
		printf("%d", A/D+1);	

	else
		printf("-1");
	
	return 0;
}
