#include <stdio.h>

#define SEAT 100

int main(void) {
	int arr[SEAT] = {0,};
	int n;
	int sam;
	int count=0;
	int i;
	
	scanf("%d", &n);
	
	for(i=0; i<n; i++) {
		scanf("%d", &sam);
		
		if(arr[sam-1]==0) arr[sam-1]++;
		else count++;
	}
	
	printf("%d", count);
	
	return 0;
}
