#include <stdio.h>

void jungR(int arr[], int n);
int min_time(int arr[], int n);

int main(void) {
	int arr[1001]={0,};
	int n;
	
	scanf("%d", &n);
	
	jungR(arr, n);
	
	printf("%d", min_time(arr, n));
	
	return 0;
}

void jungR(int arr[], int n) {
	int s;
	int i;
	
	for(i=0; i<n; i++) {
		scanf("%d", &s);
		
		arr[s]++;
	}
}

int min_time(int arr[], int n) {
	int count=0;
	int i, l;
	
	for(i=1; i<1001; i++) {
		if(arr[i]>0) {
			for(l=0; l<arr[i]; l++) {
				count += i*n;
				
				n--;
			}
		}
	}
	
	return count;
}