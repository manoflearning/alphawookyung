#include <stdio.h>

void draw_star7(int n);

int main(void) {
	int n;
	
	scanf("%d", &n);
	
	draw_star7(n);
	
	return 0;
}

void draw_star7(int n) {
	int i, l;
	
	for(i=1; i<=n; i++) {
		for(l=n-i; l>=1; l--)
			printf(" ");
		for(l=2*i-1; l>=1; l--)
			printf("*");
		printf("\n");
	}
	for(i=n-1; i>=1; i--) {
		for(l=n-i; l>=1; l--)
			printf(" ");
		for(l=2*i-1; l>=1; l--)
			printf("*");
		printf("\n");
	}
}
