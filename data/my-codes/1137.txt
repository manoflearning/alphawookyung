#include <stdio.h>

int count(int coin[], int n, int k);

int main(void) {
	int coin[10];
	int n, k;
	int i;
	
	scanf("%d %d", &n, &k);
	
	for(i=0; i<n; i++)
		scanf("%d", &coin[i]);
	
	printf("%d", count(coin, n, k));
	
	return 0;
}

int count(int coin[], int n, int k) {
	int count=0;
	int i;
	
	while(k!=0) {
		for(i=n-1; i>=0; i--)
			if(coin[i]<=k) break;
		
		k-=coin[i];
		count++;
	}
	
	return count;
}