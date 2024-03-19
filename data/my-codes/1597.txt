#include <stdio.h>

void yabawi(char s[], int t);

int main(void) {
	char s[101];
	int t;
	
	scanf("%s %d", s, &t);
	
	yabawi(s, t);
	
	return 0;
}

void yabawi(char s[], int t) {
	int a, b;
	char tmp;
	int i;
	
	for(i=0; i<t; i++) {
		scanf("%d %d", &a, &b);
		
		tmp=s[b];
		s[b]=s[a];
		s[a]=tmp;
	}
	
	printf("%s", s);
}