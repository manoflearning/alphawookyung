#include <stdio.h>

int minimum_distance(int x, int y, int w, int h) {
	int d1, d2, d3, d4;
	
	d1=x, d2=y;
	d3=w-x, d4=h-y;
	
	if(d1<=d2 && d1<=d3 && d1<=d4) return d1;
	else if(d2<=d1 && d2<=d3 && d2<=d4) return d2;
	else if(d3<=d1 && d3<=d2 && d3<=d4) return d3;
	else if(d4<=d1 && d4<=d2 && d4<=d3) return d4;
}

int main (void) {
	int x, y, w, h;
	
	scanf("%d %d %d %d", &x, &y, &w, &h);
	
	printf("%d", minimum_distance(x, y, w, h));
	
	return 0;
}
