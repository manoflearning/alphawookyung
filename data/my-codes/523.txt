#include <stdio.h>

int i, l, j;

void boonsoo_count(int X) {
	for(i=1; i<=20000; i++) {
		if(X<=i) break;
		
		X-=i;
	}
	
	if(i%2==0) printf("%d/%d", 1+(X-1), i-(X-1));
	else printf("%d/%d", i-(X-1), 1+(X-1));
}

int main (void) {
	int X;
	
	scanf("%d", &X);
	
	boonsoo_count(X);
	
	return 0;
}
