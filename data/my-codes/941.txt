#include <stdio.h>

void draw_star8(int n);

int main(void) {
	int n;
	
	scanf("%d", &n);
	
	draw_star8(n);
	
	return 0;
}

void draw_star8(n) {
	int i, l;
	
	for(i=1; l<=n; i++) {
		for(l=1; l<=i; l++)
			printf("*");
		for(l=2*n-2*i; l>=1; l--)
			printf(" ");
		for(l=1; l<=i; l++)
			printf("*");
		printf("\n");
	}
	for(i=n-1; i>=1; i--) {
		for(l=1; l<=i; l++)
			printf("*");
		for(l=2*n-2*i; l>=1; l--)
			printf(" ");
		for(l=1; l<=i; l++)
			printf("*");
		printf("\n");
	}
}
