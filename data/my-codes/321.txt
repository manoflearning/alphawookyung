#include <stdio.h>
#include <math.h>

int count(int x1, int y1, int r1, int x2, int y2, int r2) {
	if(x1==x2 && y1==y2) {
		if(r1==r2) return -1;
		
		else return 0;
	}
	else {
		if(pow(r1-r2, 2)>pow(x1-x2, 2)+pow(y1-y2, 2))
			return 0;
		else if(pow(r1-r2, 2)==pow(x1-x2, 2)+pow(y1-y2, 2))
			return 1;
		else if(pow(r1+r2, 2)>pow(x1-x2, 2)+pow(y1-y2, 2))
			return 2;
		else if(pow(r1+r2, 2)==pow(x1-x2, 2)+pow(y1-y2, 2))
			return 1;
		else
			return 0;
	}
}

int main (void) {
	int T;
	int x1, y1, r1, x2, y2, r2;
	int i;
	
	scanf("%d", &T);
	
	for(i=0; i<T; i++) {
		scanf("%d %d %d %d %d %d", &x1, &y1, &r1, &x2, &y2, &r2);
		
		printf("%d\n", count(x1, y1, r1, x2, y2, r2));
	}
	
	return 0;
}
