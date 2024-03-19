#include <stdio.h>

int climbing_tree(int a, int b, int v) {
	int count;
	
	v-=a;
	if(v%(a-b)==0) count=1+v/(a-b);
	else count=2+v/(a-b);
	
	return count;
}

int main (void) {
	int a, b, v;
	
	scanf("%d %d %d", &a, &b, &v);
	
	printf("%d", climbing_tree(a, b, v));
	
	return 0;
}