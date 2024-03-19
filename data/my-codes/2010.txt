#include <stdio.h>

void judgment(int x, int y, int z) {
	if(x*x==y*y+z*z || x*x+y*y==z*z || x*x+z*z==y*y)
		printf("right\n");
	else
		printf("wrong\n");
}

int main (void) {
	int x, y, z;
	
	while(1) {
		scanf("%d %d %d", &x, &y, &z);
		
		if(x==0 && y==0 && z==0) break;
		
		judgment(x, y, z);
	}
	
	return 0;
}