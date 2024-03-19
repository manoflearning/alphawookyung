#include <stdio.h>

int residents_number(int k, int n) {
	int r_num;
	int apart[15][15]={0,};
	int i, l, j;
	
	for(i=1; i<=14; i++)
		apart[0][i]=i;
		
	for(i=1; i<=k-1; i++)
		for(l=1; l<=n; l++)
			for(j=1; j<=l; j++)
				apart[i][l]+=apart[i-1][j];
	for(i=1; i<=n; i++)
		apart[k][n]+=apart[k-1][i];			
		
	r_num=apart[k][n];
	
	return r_num;
}

int main (void) {
	int T;
	int k, n;
	int i;
	
	scanf("%d", &T);
	
	for(i=0; i<T; i++) {
		scanf("%d %d", &k, &n);
		
		printf("%d\n", residents_number(k, n));
	}
	
	return 0;
}