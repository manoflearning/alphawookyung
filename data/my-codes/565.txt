#include <stdio.h>
int i;

int main (void) {
	int n;
	float score[1001]={0,};
	float aver;
	
	scanf("%d", &n);
	
	for(i=0; i<n; i++) {
		scanf("%f", &score[i]);
		
		if(score[n]<score[i])
			score[n]=score[i];
	}
	
	for(i=0; i<n; i++) {
		score[i] = score[i]/score[n]*100;
		
		aver += score[i];
	}
	
	printf("%.3f", (float)aver/n);
	
	return 0;
}
