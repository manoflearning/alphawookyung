#include <stdio.h>
int i, l;

int main (void) {
	int N, X;
	int t;
	
	scanf("%d %d", &N, &X);
	
	for(i=0; i<N; i++) {
		scanf("%d", &t);
		
		if(t<X)
			printf("%d ", t);
	}
	
	return 0;
}