#include <stdio.h>

void find_real_shortman(int shortman[][2]);

int main(void) {
	int shortman[9][2]={0};
	int i;
	
	for(i=0; i<9; i++)
		scanf("%d", &shortman[i][0]);
	
	find_real_shortman(shortman);
	
	for(i=0; i<9; i++)
		if(shortman[i][1])
			printf("%d\n", shortman[i][0]);
	
	return 0;
}

void find_real_shortman(int shortman[][2]) {
	int sum;
	int i, l, j;
	
	for(i=0; i<8; i++)
		for(l=i+1; l<9; l++) {
			sum=0;
			for(j=0; j<9; j++)
				if(j==i || j==l) continue;
				else sum+=shortman[j][0];
				
			if(sum==100) {
				for(j=0; j<9; j++)
					if(j==i || j==l) continue;
					else shortman[j][1]=1;
			
				i=8, l=9;
				break;
			}
		}
}