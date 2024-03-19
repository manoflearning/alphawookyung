#include <stdio.h>

int main (void) {
	int x, y;
	int a3, a4, a5, a6;
	
	scanf("%d %d", &x, &y);
	
	a3 = x * (y%10);
	a4 = x * (y%100/10);
	a5 = x * ((y-y%100)/100);
	a6 = x * y;
	
	printf("%d\n%d\n%d\n%d", a3, a4, a5, a6);
	
	return 0;
}