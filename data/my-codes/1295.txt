#include <stdio.h>

int main (void) {
	int n, i;
	int arr[1000000];
	int max, min;
	
	scanf("%d", &n);
	
	for(i=0; i<n; i++)
		scanf("%d", &arr[i]);
	
	max=arr[0], min=arr[0];
	for(i=1; i<n; i++) {
		if(arr[i]>max) max=arr[i];
		if(arr[i]<min) min=arr[i];
	}
	
	printf("%d %d", min, max);
	
	return 0;
}