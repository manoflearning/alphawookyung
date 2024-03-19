#include <stdio.h>

int count(int n) {
	int arr[246912];
	int count=0;
	int i, l;
	
	for(i=2; i<=2*n; i++)
		arr[i]=1;
		
	for(i=2; i*i<=2*n; i++)
		if(arr[i])
			for(l=2*i; l<=2*n; l+=i)
				arr[l]=0;
	
	for(i=n+1; i<=2*n; i++)
		if(arr[i]) count++;
		
	return count;
}

int main (void) {
	int n;
	
	while(1) {
		scanf("%d", &n);
		
		if(n==0) break;
		
		printf("%d\n", count(n));
	}
	
	return 0;
}