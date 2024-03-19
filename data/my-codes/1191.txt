#include <stdio.h>

int main (void) {
	char input[101];
	
	while(gets(input)) {
		if(input[strlen(input)]==EOF) break;
		
		else puts(input);
	}
	
	return 0;
}