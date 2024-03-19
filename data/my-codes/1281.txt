#include <stdio.h>

void draw_star19(int n);

int main(void) {
	int n;
	
	scanf("%d", &n);
	
	draw_star19(n);
	
	return 0;
}

void draw_star19(int n) {
	int i, l;
	int n4;
	
	n4=(n-1)*4+1;
	
	for(i=1; i<=n4/2+1; i++) {
		for(l=1; l<=n4; l++) {
			if(l==1 || l==n4)
				printf("*");
			else if(i<=l && l<=n4-i+1 && i%2==1)
				printf("*");
			else if(l%2==1 && (l<i || n4+1-l<i))
				printf("*");
			else
				printf(" ");
		}
		printf("\n");
	}
	for(i=n4/2; i>=1; i--) {
		for(l=1; l<=n4; l++) {
			if(l==1 || l==n4)
				printf("*");
			else if(i<=l && l<=n4-i+1 && i%2==1)
				printf("*");
			else if(l%2==1 && (l<i || n4+1-l<i))
				printf("*");
			else
				printf(" ");
		}
		printf("\n");
	}
}