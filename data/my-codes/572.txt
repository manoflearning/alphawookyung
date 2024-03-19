#include <stdio.h>

unsigned long long find_factor(int T, long long factor[]) {
	unsigned long long min, max, find_N;
	int i;
	
	min=factor[0], max=factor[0];
	for(i=1; i<T; i++) {
		if(min>factor[i]) min=factor[i];
		if(max<factor[i]) max=factor[i];
	}
	
	find_N=min*max;
	
	return find_N;
}

int main (void) {
	int T;
	long long factor[50];
	int i;
	
	scanf("%d", &T);
	
	for(i=0; i<T; i++)
		scanf("%llu", &factor[i]);
		
	printf("%llu", find_factor(T, factor));
	
	return 0;
}
