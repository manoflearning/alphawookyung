#include <stdio.h>

void draw_star20(int n);

int main(void) {
	int n;
	
	scanf("%d", &n);
	
	draw_star20(n);
	
	return 0;
}

void draw_star20(int n) {
	int i, l;
	
	for(i=1; i<=n; i++) {
		for(l=1; l<=2*n; l++) {
			if(i%2==1)
				if(l%2==1)
					printf("*");
				else
					printf(" ");
			else
				if(l%2==1)
					printf(" ");
				else
					printf("*");
		}
		printf("\n");
	}
}