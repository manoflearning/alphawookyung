#include <stdio.h>

int main (void) {
	int x, y;
	
	scanf("%d %d", &x, &y);
	printf("%.10f", (double)x/y);
	
	return 0;
}
