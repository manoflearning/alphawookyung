#include <stdio.h>

int main (void) {
	int N, N1, N2;
	int flag = 1;
	
	scanf("%d", &N);
	
	if(N<10)
		N1 = N * 10;
	else
		N1 = N;
		
	N2 = (N1/10 + N1%10)%10 + N%10*10;
	N1 = N2; 
	while(1) {
		if(N2 == N) {
			printf("%d", flag);
			break;
		}
		N2 = (N1/10 + N1%10)%10 + N1%10*10;
		flag++;
		N1 = N2;
	} 
	
	return 0;
}
