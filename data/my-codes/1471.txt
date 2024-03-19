#include <stdio.h>

void find_dot(int rectan[]) {
	int flag1, flag2;
	int x, y;
	int i, l;
	
	for(i=0; i<6; i+=2) {
		flag1=0, flag2=0;
		for(l=0; l<6; l+=2) {
			if(l!=i && rectan[l]!=rectan[i]) flag1++;
			if(l+1!=i+1 && rectan[l+1]!=rectan[i+1]) flag2++;
		}
		
		if(flag1==2) x=rectan[i];
		if(flag2==2) y=rectan[i+1];
	}
	
	printf("%d %d", x, y);
}

int main (void) {
	int rectan[6];
	int i;
	
	for(i=0; i<6; i+=2)
		scanf("%d %d", &rectan[i], &rectan[i+1]);
	
	find_dot(rectan);
	
	return 0;
}