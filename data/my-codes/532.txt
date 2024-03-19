#include <stdio.h>

int chessboard_coloring(char chessboard[][51], int N, int M) {
	int i, l, j, k;
	int min=64, Pmin1, Pmin2, count1, count2;
	
	for(i=0; i<=N-8; i++)
		for(l=0; l<=M-8; l++) {
			Pmin1=0, Pmin2=0, count1=0, count2=0;
			for(j=i; j<i+8; j++) {
				count1++;
				for(k=l; k<l+8; k++) {
					count2++;
					if(count1%2==1 && count2%2==0) {
						if(chessboard[j][k]=='W') Pmin1++;
						else Pmin2++;
					}
					else if(count1%2==0 && count2%2==0) {
						if(chessboard[j][k]=='B') Pmin1++;
						else Pmin2++;
					}
					else if(count1%2==0 && count2%2==1) {
						if(chessboard[j][k]=='W') Pmin1++;
						else Pmin2++;
					}
					else {
						if(chessboard[j][k]=='B') Pmin1++;
						else Pmin2++;
					}
				}
			}
			if(min>Pmin1) min=Pmin1;
			if(min>Pmin2) min=Pmin2;
		}
	
	return min;
}

int main (void) {
	int N, M;
	char chessboard[50][51];
	int i;
	
	scanf("%d %d", &N, &M);
	
	for(i=0; i<N; i++)
		scanf("%s", chessboard[i]);
	
	printf("%d", chessboard_coloring(chessboard, N, M));
	
	return 0;
}
