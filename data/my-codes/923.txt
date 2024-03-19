#include <stdio.h>
int i, l;

int main (void) {
	int T;
	
	scanf("%d", &T);
	
	for(i=0; i<T; i++) {
		for(l=0; l<=i; l++)
			printf("*");
		printf("\n");
	}
	
	return 0;
}
