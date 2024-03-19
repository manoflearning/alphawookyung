#include <stdio.h>

int main (void) {
	int score[5];
	int aver=0;
	int i;
	
	for(i=0; i<5; i++) {
		scanf("%d", &score[i]);
		
		if(score[i]<40) aver+=40;
		else aver+=score[i];
	}
	
	aver/=5;
	printf("%d", aver);
	
	return 0;
}