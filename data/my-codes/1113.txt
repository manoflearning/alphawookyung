#include <stdio.h>

void boonhae(int N) {
	int i;
	
	for(i=2; i<=N; i++) {
		if(N%i==0) {
			N/=i;
			printf("%d\n", i);
			i--;
		}
		if(i>N) break;
	}
}

int main (void) {
	int N;
	
	scanf("%d", &N);
	
	boonhae(N);
	
	return 0;
}