#include <stdio.h>

int investment(int a, int b, int c);

int main(void) {
	int t;
	int n, a, b, c;
	int sum;
	int i, l;
	
	scanf("%d", &t);
	
	for(i=0; i<t; i++) {
		scanf("%d", &n);
		
		sum=0;
		
		for(l=0; l<n; l++) {
			scanf("%d %d %d", &a, &b, &c);
			
			sum+=investment(a, b, c);
		}
		
		printf("%d\n", sum);
	}
	
	return 0;
}

int investment(int a, int b, int c) {
	int i;
	
	if(a<=0 && b<=0 && c<=0) return 0;
	else {
		int max=0;
		
		if(a>max) max=a;
		if(b>max) max=b;
		if(c>max) max=c;
		
		return max;
	}
}