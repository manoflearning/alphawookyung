#include <stdio.h>

int factorial(int n) {
	int i;
	int factorial=1;
	
	for(i=1; i<=n; i++)
		factorial*=i;
		
	return factorial;
}

int count(int n) {
	int count=0;
	int i, l, j;
	
	for(i=0; i<=n; i++)
		for(l=0; l*2<=n; l++)
			for(j=0; j*3<=n; j++)
				if(i+l*2+j*3==n) {
					count+=factorial(i+l+j)/factorial(i)/factorial(l)/factorial(j);
				}
	
	return count;
}

int main (void) {
	int T;
	int n;
	int i;
	
	scanf("%d", &T);
	
	for(i=0; i<T; i++) {
		scanf("%d", &n);
		
		printf("%d\n", count(n));
	}
	
	return 0;
}