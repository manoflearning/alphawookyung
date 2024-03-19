#include <stdio.h>
#include <string.h>

int i, l, j;

int calculate(char number[], int N) {
	int sum=0;
	
	for(i=0; i<N; i++) {
		sum+=number[i]-48;
	}
	
	return sum;
}

int main (void) {
	int N;
	char number[101];
	
	scanf("%d", &N);
	scanf("%s", number);
	
	printf("%d", calculate(number, N));
	
	return 0;
}