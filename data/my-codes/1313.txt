#include <stdio.h>

int addition();

int main(void) {
	int t;
	int i;
	
	scanf("%d", &t);
	
	for(i=0; i<t; i++)
		printf("%d\n", addition());
	
	return 0;
}

int addition() {
	int a, b;
	char c;
	
	scanf("%d %c %d", &a, &c, &b);
	
	return a+b;
}