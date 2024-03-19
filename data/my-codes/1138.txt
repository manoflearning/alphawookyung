#include <stdio.h>
#include <string.h>

void tenword_output(char input[]) {
	int i;
	
	for(i=0; i<strlen(input); i++) {
		if(i%10==0 && i!=0) printf("\n%c", input[i]);
		else printf("%c", input[i]);
	}
}

int main (void) {
	char input[101];
	
	scanf("%s", input);
	
	tenword_output(input);
	
	return 0;
}