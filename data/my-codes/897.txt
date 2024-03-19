#include <stdio.h>
int i;

int main (void) {
	int N;
	
	scanf("%d", &N);
	
	for(i=N; i>=1; i--)
		printf("%d\n", i);
	
	return 0;
}