#include <stdio.h>

int room_numbering(int H, int W, int N) {
	int r_count=1;
	
	for( ; N-H>0; ) {
		r_count++;
		N-=H;
	}

	return r_count+N*100;
}

int main (void) {
	int T;
	int H, W, N;
	int i;
	
	scanf("%d", &T);
	
	for(i=0; i<T; i++) {
		scanf("%d %d %d", &H, &W, &N);
		
		printf("%d\n", room_numbering(H, W, N));
	}
	
	return 0;
}