#include <stdio.h>
int i;

int main (void) {
	int N;
	
	scanf("%d", &N);
	
	for(i=1; i<=N; i++) {
		printf("%d\n", i);
	}
	
	return 0;
}