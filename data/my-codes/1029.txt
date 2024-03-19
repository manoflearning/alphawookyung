#include <stdio.h>
#include <string.h>

int i, l, j;

int main (void) {
	int T, R;
	char s[21];
	
	scanf("%d", &T);
	
	for(i=0; i<T; i++) {
		scanf("%d", &R);
		scanf("%s", s);
		
		for(l=0; l<strlen(s); l++)
			for(j=0; j<R; j++)	
				printf("%c", s[l]);
		printf("\n");
	}
	
	return 0;
}