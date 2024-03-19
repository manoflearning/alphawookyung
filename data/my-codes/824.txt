#include <stdio.h>

void draw_star13(int n);

int main(void) {
	int n;
	
	scanf("%d", &n);
	
	draw_star13(n);
	
	return 0;
}

void draw_star13(int n) {
	int i, l;
	
	for(i=1; i<=n; i++) {
		for(l=n-i; l<n; l++)
			printf("*");
			
		printf("\n");
	}
	for(i=n-1; i>=1; i--) {
		for(l=n-i; l<n; l++)
			printf("*");
			
		printf("\n");
	}
}
