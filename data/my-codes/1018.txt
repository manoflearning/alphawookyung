#include <stdio.h>

int nth_mineral_water(int n, int k);

int main(void) {
	int n, k;
	int answer;
	
	scanf("%d %d", &n, &k);
	
	answer = nth_mineral_water(n, k);
	printf("%d", answer);
	
	return 0;
}

int nth_mineral_water(int n, int k) {
	int count=0;
	int i;
	
	for(i=n; i>=1; i--) {
		if(n%i==0) count++;
		
		if(count==k) break;
	}
	
	if(i==0) return 0;
	else return n/i;
}
