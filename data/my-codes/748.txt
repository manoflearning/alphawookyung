#include <stdio.h>

void yut_judgement(int yut[]);

int main(void) {
	int yut[4];
	int i, l;
	
	for(i=0; i<3; i++) {
		for(l=0; l<4; l++)
			scanf("%d", &yut[l]);
		
		yut_judgement(yut);
	}
	
	return 0;
}

void yut_judgement(int yut[]) {
	int count=0;
	int i, l;
	
	for(i=0; i<4; i++)
		if(yut[i]) count++;
		
	if(count==4) printf("E\n");
	else if(count==3) printf("A\n");
	else if(count==2) printf("B\n");
	else if(count==1) printf("C\n");
	else if(count==0) printf("D\n");
}
