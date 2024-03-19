#include <stdio.h>

int differential(int c, int k);

int main(void) {
	int n, c, k;
	int count=0;
	int i;
	
	scanf("%d", &n);
	
	for(i=0; i<n; i++) {
		scanf("%d %d", &c, &k);
		
		count+=differential(c, k);
	}
	
	printf("%d", count);
	
	return 0;
}

int differential(int c, int k) {
	return c*k;
}