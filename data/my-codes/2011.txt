#include <stdio.h>
int i, l, j;

int main (void) {
	int c, n, score[1000];
	float aver, aos;
	
	scanf("%d", &c);
	
	for(i=0; i<c; i++) {
		scanf("%d", &n);
		
		aver=0, aos=0;
		for(l=0; l<n; l++) {
			scanf("%d", &score[l]);
			aver+=score[l];
		}
		aver=(float)aver/n;
		
		for(l=0; l<n; l++) {
			if(score[l]>aver) aos++;
		}
		aos=(float)aos/n*100;
		
		printf("%.3f%%\n", aos);
	}
	
	return 0;
}