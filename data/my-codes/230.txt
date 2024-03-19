#include <stdio.h>
int i;

int main (void) {
	int n, result=0;

	scanf("%d", &n);
	
	for(i=1; i<=n; i++)
		result+=i;
		
	printf("%d", result);
		
	return 0;
}