#include <stdio.h>

void draw_star5(int n);

int main(void) {
	int n;
	
	scanf("%d", &n);
	
	draw_star5(n);
	
	return 0;
}

void draw_star5(int n) {
	int i, l;
	
	for(i=1; i<=n; i++) {
		for(l=n-i; l>=1; l--)
			printf(" ");
		for(l=i*2-1; l>=1; l--)
			printf("*");
		printf("\n");
	}
}
