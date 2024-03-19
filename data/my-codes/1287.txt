#include <stdio.h>
#include <string.h>

int i, l, j;

int main (void) {
	char word[101];
	
	scanf("%s", word);
	
	for(i='a'; i<='z'; i++)
		for(l=0; l<strlen(word); l++) {
			if(i==word[l]) {
				printf("%d ", l);
				break;
			}
			if(l==strlen(word)-1)
				printf("-1 ");
		}
	
	return 0;
}