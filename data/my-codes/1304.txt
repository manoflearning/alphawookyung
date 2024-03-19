#include <stdio.h>

void draw_star15(int n);

int main(void) {
	int n;
	
	scanf("%d", &n);
	
	draw_star15(n);
	
	return 0;
}

void draw_star15(int n) {
	int i, l;
	if(n!=1) {
		for(i=1; i<n; i++)
			printf(" ");
		printf("*\n");
		for(i=2; i<=n-1; i++) {
			for(l=i; l<n; l++)
				printf(" ");
			printf("*");
			for(l=2*(i-2)+1; l>=1; l--)
				printf(" ");
			printf("*\n");
		}
		for(i=1; i<=2*n-1; i++)
			printf("*");
	}
	else
		printf("*");
}