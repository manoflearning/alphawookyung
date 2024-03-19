#include <stdio.h>

int i, l, j;

int hansu(int N) {
	int digit[4];
	int fake_i;
	int count=0;
	
	for(i=1; i<=N; i++) {
		fake_i=i;
		for(l=0; fake_i!=0; l++) {
			digit[l] = fake_i%10;
			fake_i/=10;
		}
		
		if(l==1) count++;
		if(l==2) count++;
		if(l==3 && digit[0]-digit[1]==digit[1]-digit[2])
			count++;
		if(l==4)
			if(digit[0]-digit[1]==digit[1]-digit[2]==digit[2]-digit[3])
				count++;
		if(l==5)
			if(digit[0]-digit[1]==digit[1]-digit[2]==digit[2]-digit[3]==digit[3]-digit[4])
				count++;
	}
		
	return count;
}

int main (void) {
	int N;
	
	scanf("%d", &N);
	
	printf("%d", hansu(N));
	
	return 0;
}
