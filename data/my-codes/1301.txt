#include <stdio.h>

int main (void) {
	int t, m, n;
	int i;
	
	scanf("%d", &t);
	
	for(i=0; i<t; i++) {
		scanf("%d %d", &m, &n);
		
		printf("%d\n", m+n);
	}
	
	return 0;
}