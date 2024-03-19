#include <stdio.h>

void draw_star6(int n);

int main (void) {
	int n;
	
	scanf("%d", &n);
	
	draw_star6(n);
	
	return 0;
}

void draw_star6(int n) {
	int i, l;
	
	for(i=1; i<=n; i++) {
		for(l=n-i+2; l<=n; l++)
			printf(" ");
		for(l=2*n+1-2*i; l>=1; l--)
			printf("*");
			
		printf("\n");
	}
}
